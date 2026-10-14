#include "oblique_apply_projection.cuh"
#include "parallel_utils.hpp"
#include "yggdrasil_decision_forests/learner/decision_tree/maximize_groupings.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <thread>

#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/random.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

ABSL_FLAG(std::string, input_mode, "csv",
          "Data input mode: 'csv', 'trunk' (synthetic), or 'tfrecord'.");
ABSL_FLAG(std::string, train_csv, "",
          "Path to training CSV file.");
ABSL_FLAG(std::string, ds_path, "",
          "Path (without extension) to TF-Record file (for tfrecord mode).");
ABSL_FLAG(std::string, label_col, "target",
          "Name of label column (used in all modes).");

ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(int, num_trees, 1, "Number of trees in the random forest.");
ABSL_FLAG(int, tree_depth, 6, "Maximum depth of trees (-1 for unlimited).");
ABSL_FLAG(std::string, sweep_tree_depths, "",
          "Comma-separated tree depths to sweep (overrides --tree_depth).");
ABSL_FLAG(std::string, sweep_density_factors, "",
          "Comma-separated projection_density_factors to sweep (overrides --projection_density_factor).");
ABSL_FLAG(std::string, sweep_results_csv, "",
          "Path to write sweep results CSV (one row per run).");

ABSL_FLAG(std::string, feature_split_type, "Oblique",
          "Type of feature splits in decision trees: 'Axis Aligned' or 'Oblique'.");

// Oblique split parameters (only used when feature_split_type = "Oblique")
ABSL_FLAG(int, max_num_projections, 1000,
          "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, num_projections_exponent, .5,
          "Exponent to determine number of projections.");
ABSL_FLAG(float, projection_density_factor, 1.5f,
          "Projection density factor.");

ABSL_FLAG(std::string, ensemble_method, "Bagging",
          "Ensemble method: 'Bagging' (Random Forest) or 'Boosting' (Gradient Boosted Trees/MART).");
ABSL_FLAG(float, shrinkage, 0.1f,
          "Learning rate for boosting (only used when ensemble_method = 'Boosting').");         

ABSL_FLAG(bool, bootstrap_training_dataset, true,
          "Whether to use bootstrap sampling of the training dataset (Bagging only). NOTE: NO BOOTSTRAP USES ~50% MORE MEMORY. POSSIBLE BUG");
ABSL_FLAG(bool, compute_oob_performances, false,
          "Whether to compute out-of-bag performances (only for csv mode).");

ABSL_FLAG(std::string, growing_strategy, "Local",
          "Type of Tree Growing Strategy: 'Local' - depth-first using NodeTrain or 'GlobalBestFirst' - PriorityQueue the nodes based on Score() Gain.");

ABSL_FLAG(uint32_t, seed, 1,
          "PRNG seed (for deterministic synthetic mode and model training).");

// Histogram-based splits - Updated to match Yggdrasil implementation
ABSL_FLAG(std::string, numerical_split_type, "Random",
          "Type of numerical splitting: 'Exact', 'Random', or 'Equal Width'.");
ABSL_FLAG(int, histogram_num_bins, 256,
          "Number of bins for histogram splitting.");

ABSL_FLAG(std::string, model_out_dir, "",
          "Path to output trained model directory (optional)."
          " If empty, model is not saved.");

ABSL_FLAG(bool, csv_single_pass_load, true,
          "Load CSV in a single file scan instead of YDF's default two reads.");

// Synthetic mode flags
ABSL_FLAG(int64_t, rows, 4096, "Number of examples (for synthetic mode).");
ABSL_FLAG(int, cols, 4096, "Number of numerical features (for synthetic mode).");

// Dataset size sweep flags (synthetic mode only)
ABSL_FLAG(std::string, sweep_rows, "",
          "Comma-separated row counts to sweep (overrides --rows). Synthetic mode only.");
ABSL_FLAG(std::string, sweep_cols, "",
          "Comma-separated column counts to sweep (overrides --cols). Synthetic mode only.");

namespace single_pass_csv {

using namespace yggdrasil_decision_forests;

std::string FormatLabelKey(float v) {
  if (std::isfinite(v) && std::floor(v) == v && std::fabs(v) < 1e15f)
    return absl::StrCat(static_cast<int64_t>(v));
  return absl::StrCat(v);
}

absl::StatusOr<std::vector<std::string>> ReadHeader(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open())
    return absl::NotFoundError(absl::StrCat("Cannot open CSV: ", path));
  std::string line;
  if (!std::getline(in, line))
    return absl::InvalidArgumentError(absl::StrCat("Empty CSV: ", path));
  if (!line.empty() && line.back() == '\r') line.pop_back();
  std::vector<std::string> names = absl::StrSplit(line, ',');
  for (auto& n : names) n = std::string(absl::StripAsciiWhitespace(n));
  if (names.empty())
    return absl::InvalidArgumentError("CSV header has no columns");
  return names;
}

absl::StatusOr<std::unique_ptr<dataset::VerticalDataset>> Load(
    const std::string& csv_path, const std::string& label_col) {
  auto header_or = ReadHeader(csv_path);
  if (!header_or.ok()) return header_or.status();
  const std::vector<std::string>& header = header_or.value();

  int label_idx = -1;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) {
    if (header[i] == label_col) { label_idx = i; break; }
  }
  if (label_idx < 0)
    return absl::InvalidArgumentError(absl::StrCat(
        "Label column '", label_col, "' not found in CSV header"));

  dataset::proto::DataSpecification numeric_spec;
  for (const auto& name : header) {
    auto* col = numeric_spec.add_columns();
    col->set_name(name);
    col->set_type(dataset::proto::ColumnType::NUMERICAL);
  }
  auto ds = std::make_unique<dataset::VerticalDataset>();
  const absl::Status load_status =
      dataset::LoadVerticalDataset("csv:" + csv_path, numeric_spec, ds.get());
  if (!load_status.ok()) return load_status;

  const int64_t nrow = ds->nrow();
  if (nrow <= 0) return absl::InvalidArgumentError("CSV has no data rows");

  auto label_num_or = ds->numerical_column(label_idx);
  if (!label_num_or.ok()) return label_num_or.status();
  const std::vector<float> label_vals = label_num_or.value()->values();

  std::map<float, int64_t> counts;
  int64_t label_nas = 0;
  for (float v : label_vals) {
    if (std::isnan(v)) { ++label_nas; continue; }
    ++counts[v];
  }
  if (counts.size() < 2)
    return absl::FailedPreconditionError(
        "Label has fewer than 2 distinct values");

  std::vector<std::pair<float, int64_t>> vocab(counts.begin(), counts.end());
  std::stable_sort(vocab.begin(), vocab.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });

  dataset::proto::Column cat_spec;
  cat_spec.set_name(header[label_idx]);
  cat_spec.set_type(dataset::proto::ColumnType::CATEGORICAL);
  auto* cat = cat_spec.mutable_categorical();
  cat->set_is_already_integerized(false);
  cat->set_number_of_unique_values(static_cast<int64_t>(vocab.size()) + 1);
  auto& items = *cat->mutable_items();
  {
    auto& ood = items[dataset::kOutOfDictionaryItemKey];
    ood.set_index(dataset::kOutOfDictionaryItemIndex);
    ood.set_count(label_nas);
  }
  std::map<float, int32_t> value_to_index;
  int64_t most_freq_idx = dataset::kOutOfDictionaryItemIndex;
  int64_t most_freq_count = -1;
  for (size_t i = 0; i < vocab.size(); ++i) {
    const int32_t idx = static_cast<int32_t>(i) + 1;
    auto& item = items[FormatLabelKey(vocab[i].first)];
    item.set_index(idx);
    item.set_count(vocab[i].second);
    value_to_index[vocab[i].first] = idx;
    if (vocab[i].second > most_freq_count) {
      most_freq_count = vocab[i].second;
      most_freq_idx = idx;
    }
  }
  cat->set_most_frequent_value(most_freq_idx);

  auto replaced = ds->ReplaceColumn(label_idx, cat_spec);
  if (!replaced.ok()) return replaced.status();
  auto cat_col_or = ds->mutable_categorical_column(label_idx);
  if (!cat_col_or.ok()) return cat_col_or.status();
  auto* cat_col = cat_col_or.value();
  for (int64_t r = 0; r < nrow; ++r) {
    const float v = label_vals[r];
    if (std::isnan(v))
      cat_col->Set(r, dataset::VerticalDataset::CategoricalColumn::kNaValue);
    else
      cat_col->Set(r, value_to_index[v]);
  }

  for (int col_idx = 0; col_idx < ds->data_spec().columns_size(); ++col_idx) {
    if (col_idx == label_idx) continue;
    auto* col_spec = ds->mutable_data_spec()->mutable_columns(col_idx);
    if (col_spec->type() != dataset::proto::ColumnType::NUMERICAL) continue;
    auto num_or = ds->numerical_column(col_idx);
    if (!num_or.ok()) continue;
    const std::vector<float>& vals = num_or.value()->values();
    double sum = 0.0, sum_sq = 0.0;
    int64_t count = 0, nas = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : vals) {
      if (std::isnan(v)) { ++nas; continue; }
      sum += v;
      sum_sq += static_cast<double>(v) * v;
      ++count;
      min_v = std::min(min_v, v);
      max_v = std::max(max_v, v);
    }
    auto* num = col_spec->mutable_numerical();
    if (count > 0) {
      const double mean = sum / count;
      double var = sum_sq / count - mean * mean;
      if (var < 0) var = 0;
      num->set_mean(mean);
      num->set_standard_deviation(std::sqrt(var));
      num->set_min_value(min_v);
      num->set_max_value(max_v);
    }
    col_spec->set_count_nas(nas);
  }

  ds->mutable_data_spec()->set_created_num_rows(nrow);
  return ds;
}

}  // namespace single_pass_csv

namespace synthetic {

using namespace yggdrasil_decision_forests;

dataset::proto::DataSpecification MakeSyntheticSpec(
    int cols, int64_t rows, int label_mod, const std::string& label_col) {
  dataset::proto::DataSpecification spec;
  for (int c = 0; c < cols; ++c) {
    auto* f = spec.add_columns();
    f->set_name("x" + std::to_string(c));
    f->set_type(dataset::proto::NUMERICAL);
  }
  auto* lbl = spec.add_columns();
  lbl->set_name(label_col);
  lbl->set_type(dataset::proto::CATEGORICAL);
  lbl->mutable_categorical()->set_number_of_unique_values(3);
  lbl->mutable_categorical()->set_is_already_integerized(true);
  spec.set_created_num_rows(rows);
  return spec;
}

dataset::VerticalDataset MakeTrunkDataset(
    const dataset::proto::DataSpecification& spec,
    int64_t rows, int cols, uint32_t seed) {
  dataset::VerticalDataset ds;
  ds.set_data_spec(spec);
  CHECK_OK(ds.CreateColumnsFromDataspec());
  ds.Resize(rows);
  using RNG = std::minstd_rand;

  constexpr int kNInformative = 256;
  const int ninform = std::min(kNInformative, cols);

  std::vector<float> mu0(cols, 0.f), mu1(cols, 0.f);
  for (int j = 0; j < ninform; ++j) {
    const float f = 1.f / std::sqrt(float(j + 1));
    mu0[j] = -f;
    mu1[j] = f;
  }

  for (int j = 0; j < cols; ++j) {
    std::seed_seq seq{seed, static_cast<uint32_t>(j)};
    RNG rng(seq);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    auto* col = ds.MutableColumnWithCast<
        dataset::VerticalDataset::NumericalColumn>(j)->mutable_values();

    for (int64_t i = 0; i < rows; ++i) {
      const bool cls1 = (i >= rows / 2);
      const float mean = cls1 ? mu1[j] : mu0[j];
      (*col)[i] = mean + normal(rng);
    }
  }

  auto* y = ds.MutableColumnWithCast<
      dataset::VerticalDataset::CategoricalColumn>(cols)->mutable_values();
  for (int64_t i = 0; i < rows; ++i)
    (*y)[i] = (i >= rows / 2) ? 2 : 1;

  return ds;
}

}  // namespace synthetic

struct PerTreeTiming {
  int tree_id;
  double ap_kernel_ms;
  double tree_total_ms;
  double sampling_ms;
  double hist_ms;
  double free_ms;
  double split_ms;
  double node_train_ms;
  double drsw_setup_ms;
  double drsw_proj_ms;
  double compute_offsets_ms;  // drsw_depth_ms: NODE_ID_DRSW ComputeOffsets
  double setup_ms;            // one-time GPU setup before the BFS loop
  double gpu_cleanup_ms;      // end-of-tree cudaFree
  double partition_free_ms;   // frees inside the split step
};

std::vector<PerTreeTiming> ParsePerDepthTimingCSV(
    const std::string& path, int first_tree_id) {
  std::vector<PerTreeTiming> results;
  std::ifstream in(path);
  if (!in.is_open()) return results;
  std::string line;
  std::getline(in, line);  // skip header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> fields = absl::StrSplit(line, ',');
    // Need all 26 columns through drsw_proj_ms (field 25).
    if (fields.size() < 26) continue;
    // TOTAL rows have "TOTAL" in the depth column (field 4)
    if (absl::StripAsciiWhitespace(fields[4]) != "TOTAL") continue;
    int tree_id = 0;
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(fields[3]), &tree_id))
      continue;
    if (tree_id < first_tree_id) continue;
    PerTreeTiming t;
    t.tree_id = tree_id;
    float v;
    // Column indices MUST match the CSV header written in training.cc:
    //  0 max_depth       1 density_factor  2 dedup_ms       3 tree
    //  4 depth           5 nodes           6 rows           7 wall_ms
    //  8 sample_proj_ms  9 flatten_sel_ms 10 flatten_proj  11 host_to_device_ms
    // 12 drsw_depth_ms  13 apply_proj_kernel_ms            14 prefix_sum_ms
    // 15 eval_proj_ms   16 partition_ms   17 node_train_ms 18 sections_sum_ms
    // 19 tree_total_ms  20 setup_ms       21 gpu_cleanup_ms 22 sampling_total_ms
    // 23 partition_free_ms  24 drsw_setup_ms  25 drsw_proj_ms
    t.ap_kernel_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[13]), &v) ? v : 0;
    t.tree_total_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[19]), &v) ? v : 0;
    t.sampling_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[22]), &v) ? v : 0;
    t.hist_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[15]), &v) ? v : 0;
    double sad_free = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[23]), &v) ? v : 0;
    double cleanup = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[21]), &v) ? v : 0;
    t.free_ms = sad_free + cleanup;
    t.partition_free_ms = sad_free;
    t.gpu_cleanup_ms = cleanup;
    t.setup_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[20]), &v) ? v : 0;
    t.split_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[16]), &v) ? v : 0;
    t.node_train_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[17]), &v) ? v : 0;
    t.drsw_setup_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[24]), &v) ? v : 0;
    t.drsw_proj_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[25]), &v) ? v : 0;
    t.compute_offsets_ms = absl::SimpleAtof(absl::StripAsciiWhitespace(fields[12]), &v) ? v : 0;
    results.push_back(t);
  }
  return results;
}

template<class DataSet>
void runExperiment(const DataSet& dataset, const Utils::ExperimentParams& params, int exp_id)
{
    // Allocate device memory for flattened data and labels
    float* d_flat_data = nullptr;
    cudaMalloc(&d_flat_data, static_cast<size_t>(dataset.num_rows) * dataset.num_cols * sizeof(float));
    cudaMemcpy(d_flat_data, dataset.flattened.data(), static_cast<size_t>(dataset.num_rows) * dataset.num_cols * sizeof(float), cudaMemcpyHostToDevice);
    int* d_labels = nullptr;
    cudaMalloc(&d_labels, dataset.num_rows * sizeof(int));
    cudaMemcpy(d_labels, dataset.labels.data(), dataset.num_rows * sizeof(int), cudaMemcpyHostToDevice);

    auto split = Utils::splitExamples(dataset.num_rows, params.num_nodes, 12345, /*sort_chunks=*/true);
    // split.splits[i]         — indices for node i
    // split.selected_examples — all indices shuffled into the nodes / sorted or not sorted per node
    // split.node_start_idx    — start offsets into the flat shuffled array
    // split.node_counts       — count per node
    // split.node_ids          — node id for each example
    // split.node_offsets      — offset within the node for each example
    // split.node_row_off       — start offsets for each node in the flat shuffled array
    // split.node_row_start_by_row — start offset for each row in the flat shuffled array
    // split.node_counts_by_row — count of examples in the node for each row
    // split.max_rows_per_node  — maximum number of rows per node
    
    uint64_t base_seed = 12345;
    auto proj = Utils::generateProjections(params.num_proj,
                                           params.num_nodes,
                                           dataset.num_cols,
                                           params.selected_features_count,
                                           base_seed);

    // std::cout << "col_idx before reorder:\n";
    // for (int n = 0; n < params.num_nodes; ++n) {
    //     std::cout << "node " << n << ":";
    //     for (int p = 0; p < params.num_proj; ++p)
    //         for (int i = 0; i < params.selected_features_count; ++i)
    //             std::cout << " " << proj.col_idx[n][p][i];
    //     std::cout << "\n";
    // }

    Utils::reorderProjections(proj, params.num_proj, params.num_nodes,
                              params.selected_features_count,
                              params.reorder_strategy, params.verbose);

    // std::cout << "col_idx after reorder:\n";
    // for (int n = 0; n < params.num_nodes; ++n) {
    //     std::cout << "node " << n << ":";
    //     for (int p = 0; p < params.num_proj; ++p)
    //         for (int i = 0; i < params.selected_features_count; ++i)
    //             std::cout << " " << proj.col_idx[n][p][i]; //3D vector: node -> projection -> list of column indices
    //     std::cout << "\n";
    // }

    // Flatten the projections for GPU processing
    auto flat = Utils::flattenProjections(proj, params.num_proj, params.num_nodes);
    // flat.col_idx, flat.weights
    // flat.offsets for each segment (projection-node pair) in the flattened arrays

    int* d_selected_examples = nullptr;
    float* d_col_add_projected = nullptr;
    int* d_node_ids = nullptr;
    int* d_node_offsets = nullptr;
    int* d_node_row_off = nullptr;

    int* d_node_row_start_by_row = nullptr;
    int* d_node_counts = nullptr;

    cudaMalloc(&d_selected_examples, static_cast<size_t>(split.selected_examples.size()) * sizeof(int)); 
    cudaMalloc(&d_col_add_projected, static_cast<size_t>(split.selected_examples.size()) * static_cast<size_t>(params.num_proj) * sizeof(float)); //num_rows * num_proj
    cudaMalloc(&d_node_ids, static_cast<size_t>(dataset.num_rows) * sizeof(int));
    cudaMalloc(&d_node_offsets, static_cast<size_t>(dataset.num_rows) * sizeof(int));
    cudaMalloc(&d_node_row_off, split.node_row_off.size()*sizeof(int));

    cudaMalloc(&d_node_row_start_by_row, dataset.num_rows * sizeof(int));
    cudaMalloc(&d_node_counts, params.num_nodes * sizeof(int));

    cudaMemcpy(d_selected_examples, split.selected_examples.data(), static_cast<size_t>(split.selected_examples.size()) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_ids, split.node_ids.data(), static_cast<size_t>(dataset.num_rows) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_offsets, split.node_offsets.data(), static_cast<size_t>(dataset.num_rows) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_row_off, split.node_row_off.data(), split.node_row_off.size()*sizeof(int), cudaMemcpyHostToDevice);
   
    cudaMemcpy(d_node_row_start_by_row, split.node_row_start_by_row.data(), dataset.num_rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_counts, split.node_counts.data(), params.num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    

    int* d_offsets = nullptr; //Offsets for start of each segment's columns in the flat arrays
    int* d_flat_projection_col_idx = nullptr;
    float* d_flat_projection_weights = nullptr;
    int* d_flat_projection_col_idx_shared = nullptr;
    float* d_flat_projection_weights_shared = nullptr;

    cudaMalloc((void **)&d_offsets, flat.offsets.size() * sizeof(int)); 
    cudaMalloc((void **)&d_flat_projection_col_idx, flat.col_idx.size() * sizeof(int)); 
    cudaMalloc((void **)&d_flat_projection_weights, flat.weights.size() * sizeof(float)); 
    cudaMalloc((void **)&d_flat_projection_col_idx_shared, params.num_proj * params.selected_features_count * sizeof(int));
    cudaMalloc((void **)&d_flat_projection_weights_shared, params.num_proj * params.selected_features_count * sizeof(float));

    cudaMemcpy(d_offsets, flat.offsets.data(), flat.offsets.size() * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_flat_projection_col_idx, flat.col_idx.data(), flat.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_flat_projection_weights, flat.weights.data(), flat.weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flat_projection_col_idx_shared, flat.col_idx.data(), params.num_proj * params.selected_features_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flat_projection_weights_shared, flat.weights.data(), params.num_proj * params.selected_features_count * sizeof(float), cudaMemcpyHostToDevice);


    const int num_seg = params.num_nodes * params.num_proj;
    
    std::vector<float> h_min(num_seg,  FLT_MAX);   // or std::numeric_limits<float>::infinity()
    std::vector<float> h_max(num_seg, -FLT_MAX);   // or -std::numeric_limits<float>::infinity()     
    float* d_min_vals = nullptr;
    float* d_max_vals = nullptr;  
    cudaMalloc(&d_min_vals, num_seg * sizeof(float));
    cudaMalloc(&d_max_vals, num_seg * sizeof(float));

    
    for (int iter = 0; iter < params.num_iters; ++iter) {
        std::cout << "=== Iteration " << iter << " ===" << std::endl;
       
        // Initialize min/max buffers for histogramming (one per segment) per iteration since they are updated in-place in the AP kernel for demo purposes (instead of doing a separate kernel to find min/max after AP)
        cudaMemcpy(d_min_vals, h_min.data(), num_seg * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_vals, h_max.data(), num_seg * sizeof(float), cudaMemcpyHostToDevice);

        ApplyProjectionBaseline(
            d_flat_data,                     // your original device dataset
            d_selected_examples,             // flat rows (all nodes)
            d_col_add_projected,             // output buffer (device)
            d_node_ids,
            d_node_offsets,
            d_node_counts,
            d_node_row_start_by_row,
            params.num_proj,
            dataset.num_rows, //Total rows in original dataset
            num_seg,
            params.num_nodes,
            params.num_elems_per_thread,
            split.node_start_idx,
            split.node_counts,
            params.selected_features_count,
            params.alternate,
            d_min_vals,
            d_max_vals,
            split.max_rows_per_node,
            split.node_row_off.data(),
            d_node_row_off,
            d_offsets,
            d_flat_projection_col_idx,
            d_flat_projection_weights,
            d_flat_projection_col_idx_shared,
            d_flat_projection_weights_shared,
            params.verbose);
        cudaDeviceSynchronize();
    }

    // --- CPU reference test ---
    int total_selected = static_cast<int>(split.selected_examples.size());
    std::size_t proj_size = static_cast<std::size_t>(total_selected) * params.num_proj;
    std::vector<float> h_projected(proj_size);
    cudaMemcpy(h_projected.data(), d_col_add_projected, proj_size * sizeof(float), cudaMemcpyDeviceToHost);

    Utils::verifyProjection(h_projected.data(), proj, dataset.flattened,
                            split.selected_examples, split.node_row_off,
                            params.num_proj, params.num_nodes,
                            dataset.num_rows, params.selected_features_count);

    cudaFree(d_flat_data);
    cudaFree(d_labels); 
    cudaFree(d_selected_examples);
    cudaFree(d_col_add_projected);
    cudaFree(d_node_ids);
    cudaFree(d_node_offsets);
    cudaFree(d_node_row_off);
    cudaFree(d_min_vals);
    cudaFree(d_max_vals);
    cudaFree(d_offsets);
    cudaFree(d_flat_projection_col_idx);
    cudaFree(d_flat_projection_weights);
    cudaFree(d_flat_projection_col_idx_shared);
    cudaFree(d_flat_projection_weights_shared);
    cudaFree(d_node_row_start_by_row);
    cudaFree(d_node_counts);    
}


int main (int argc, char* argv[]) {

    absl::ParseCommandLine(argc, argv);
    const std::string csv_path = absl::GetFlag(FLAGS_train_csv);
    const auto mode = absl::GetFlag(FLAGS_input_mode);

    // Validate required input_mode flag
    if (mode.empty()) {
        std::cerr << "Error: --input_mode is required. Use csv, synthetic, or tfrecord.\n";
        return 1;
    }

    // 1) Build parameter grids
    std::vector<int> depths_to_sweep;
    std::vector<float> densities_to_sweep;

    const std::string sweep_depths_str = absl::GetFlag(FLAGS_sweep_tree_depths);
    const std::string sweep_densities_str = absl::GetFlag(FLAGS_sweep_density_factors);

    if (!sweep_depths_str.empty()) {
        for (const auto& s : absl::StrSplit(sweep_depths_str, ',', absl::SkipWhitespace())) {
            int v;
            if (absl::SimpleAtoi(s, &v)) depths_to_sweep.push_back(v);
        }
    }
    if (depths_to_sweep.empty()) depths_to_sweep.push_back(absl::GetFlag(FLAGS_tree_depth));

    if (!sweep_densities_str.empty()) {
        for (const auto& s : absl::StrSplit(sweep_densities_str, ',', absl::SkipWhitespace())) {
            float v;
            if (absl::SimpleAtof(s, &v)) densities_to_sweep.push_back(v);
        }
    }
    if (densities_to_sweep.empty()) densities_to_sweep.push_back(absl::GetFlag(FLAGS_projection_density_factor));

    // Dataset size sweep (synthetic mode only)
    std::vector<int64_t> rows_to_sweep;
    std::vector<int> cols_to_sweep;
    const std::string sweep_rows_str = absl::GetFlag(FLAGS_sweep_rows);
    const std::string sweep_cols_str = absl::GetFlag(FLAGS_sweep_cols);

    if (!sweep_rows_str.empty()) {
        for (const auto& s : absl::StrSplit(sweep_rows_str, ',', absl::SkipWhitespace())) {
            int64_t v;
            if (absl::SimpleAtoi(s, &v)) rows_to_sweep.push_back(v);
        }
    }
    if (rows_to_sweep.empty()) rows_to_sweep.push_back(absl::GetFlag(FLAGS_rows));

    if (!sweep_cols_str.empty()) {
        for (const auto& s : absl::StrSplit(sweep_cols_str, ',', absl::SkipWhitespace())) {
            int v;
            if (absl::SimpleAtoi(s, &v)) cols_to_sweep.push_back(v);
        }
    }
    if (cols_to_sweep.empty()) cols_to_sweep.push_back(absl::GetFlag(FLAGS_cols));

    // Pre-parse flags that stay constant across the grid
    const int num_threads_flag = absl::GetFlag(FLAGS_num_threads);
    const std::string ensemble_method = absl::GetFlag(FLAGS_ensemble_method);
    const auto growing_strategy = absl::GetFlag(FLAGS_growing_strategy);
    const std::string feature_split_type = absl::GetFlag(FLAGS_feature_split_type);
    const std::string hist_type = absl::GetFlag(FLAGS_numerical_split_type);
    const std::string base_out_dir = absl::GetFlag(FLAGS_model_out_dir);
    const uint32_t seed = absl::GetFlag(FLAGS_seed);
    const int num_trees = absl::GetFlag(FLAGS_num_trees);

    LOG(INFO) << "Sweep grid: " << rows_to_sweep.size() << " row_counts x "
              << cols_to_sweep.size() << " col_counts x "
              << depths_to_sweep.size() << " depths x "
              << densities_to_sweep.size() << " densities = "
              << rows_to_sweep.size() * cols_to_sweep.size() *
                 depths_to_sweep.size() * densities_to_sweep.size() << " runs";

    const std::string results_csv_path = absl::GetFlag(FLAGS_sweep_results_csv);
    std::ofstream results_csv;
    if (!results_csv_path.empty()) {
        results_csv.open(results_csv_path);
        results_csv << "run,rows,cols,tree_depth,projection_density_factor,"
                    << "num_trees,train_time_s,"
                    // canonical buckets (sum == tree_total)
                    << "sampling_ms,projection_ms,split_finding_ms,example_mgmt_ms,"
                    << "node_train_ms,overhead_ms,tree_total_ms,"
                    // raw drill-down
                    << "compute_offsets_ms,partition_ms,apply_proj_kernel_ms,"
                    << "drsw_setup_ms,drsw_proj_ms,eval_proj_ms,free_ms" << std::endl;
    }

    // Per-tree totals (one row per tree), separate from the per-depth CSV.
    std::ofstream per_tree_csv("per_tree_timing.csv");
    if (per_tree_csv.is_open()) {
        per_tree_csv << "run,tree,"
                     // canonical buckets (sum == tree_total)
                     << "sampling_ms,projection_ms,split_finding_ms,example_mgmt_ms,"
                     << "node_train_ms,overhead_ms,tree_total_ms,"
                     // example_mgmt sub-parts
                     << "compute_offsets_ms,drsw_setup_ms,partition_ms,"
                     // overhead sub-parts
                     << "setup_ms,gpu_cleanup_ms,partition_free_ms,overhead_gap_ms,"
                     // other raw
                     << "apply_proj_kernel_ms,drsw_proj_ms,eval_proj_ms\n";
    }

    // Track csv_tree_id across runs (training.cc uses a static counter that
    // persists across TrainWithStatus calls within the same process).
    int next_tree_id = 0;
    int run_idx = 0;

    for (int64_t sweep_rows : rows_to_sweep) {
    for (int sweep_cols : cols_to_sweep) {

    // 2) Prepare data source (regenerate for each size in trunk mode)
    yggdrasil_decision_forests::dataset::proto::DataSpecification data_spec;
    std::unique_ptr<yggdrasil_decision_forests::dataset::VerticalDataset> tf_ds;
    yggdrasil_decision_forests::dataset::VerticalDataset* ds_ptr = nullptr;
    std::string label_col;

    if (mode == "csv") {

        label_col = absl::GetFlag(FLAGS_label_col);

        if (csv_path.empty()) {
            std::cerr << "Error: --train_csv is required in csv mode.\n";
            return 1;
        }
        if (label_col.empty()) {
            std::cerr << "Error: --label_col is required in csv mode.\n";
            return 1;
        }

        bool single_pass_ok = false;
        if (absl::GetFlag(FLAGS_csv_single_pass_load)) {
            LOG(INFO) << "Loading CSV in a single pass: " << csv_path;
            auto ds_or = single_pass_csv::Load(csv_path, label_col);
            if (ds_or.ok()) {
                tf_ds = std::move(ds_or.value());
                ds_ptr = tf_ds.get();
                data_spec = tf_ds->data_spec();
                single_pass_ok = true;
                LOG(INFO) << "Single-pass CSV load complete: " << tf_ds->nrow()
                          << " rows, " << data_spec.columns_size() << " columns.";
            } else {
                LOG(WARNING) << "Single-pass CSV load not applicable ("
                             << ds_or.status().message()
                             << "); falling back to two-pass inference + load.";
            }
        }

        if (!single_pass_ok) {
            LOG(INFO) << "Inferring DataSpec from CSV: " << csv_path;
            yggdrasil_decision_forests::dataset::proto::DataSpecificationGuide guide;
            auto* col_guide = guide.add_column_guides();
            col_guide->set_column_name_pattern(label_col);
            col_guide->set_type(yggdrasil_decision_forests::dataset::proto::ColumnType::CATEGORICAL);

            yggdrasil_decision_forests::dataset::CreateDataSpec(
                "csv:" + csv_path,
                /*require_same_dataset_fields=*/false,
                guide,
                &data_spec);

            tf_ds = std::make_unique<yggdrasil_decision_forests::dataset::VerticalDataset>();
            CHECK_OK(yggdrasil_decision_forests::dataset::LoadVerticalDataset(
                "csv:" + csv_path, data_spec, tf_ds.get()));
            ds_ptr = tf_ds.get();
            LOG(INFO) << "Loaded " << tf_ds->nrow() << " rows, "
                      << data_spec.columns_size() << " columns.";
        }
    } else if (mode == "tfrecord") {
        label_col = absl::GetFlag(FLAGS_label_col);
        const std::string path = absl::GetFlag(FLAGS_ds_path);
        if (path.empty()) {
            std::cerr << "Error: --ds_path is required in tfrecord mode.\n";
            return 1;
        }
        LOG(INFO) << "Loading TFRecord dataset from: " << path;
        CHECK_OK(file::GetTextProto(
            path + ".data_spec.pb", &data_spec,
            file::Defaults()));
        tf_ds = std::make_unique<yggdrasil_decision_forests::dataset::VerticalDataset>();
        CHECK_OK(yggdrasil_decision_forests::dataset::LoadVerticalDataset(
            "tfrecord:" + path, data_spec, tf_ds.get()));
        ds_ptr = tf_ds.get();
        sweep_rows = tf_ds->nrow();
        sweep_cols = data_spec.columns_size() - 1;
        LOG(INFO) << "Loaded TFRecord: " << sweep_rows << " rows, "
                  << sweep_cols << " feature columns.";
    } else if (mode == "trunk") {
        label_col = "y";

        LOG(INFO) << "Generating trunk synthetic dataset: rows=" << sweep_rows
                  << ", cols=" << sweep_cols << ", seed=" << seed;

        data_spec = synthetic::MakeSyntheticSpec(sweep_cols, sweep_rows, 2, label_col);
        auto ds = synthetic::MakeTrunkDataset(data_spec, sweep_rows, sweep_cols, seed);
        tf_ds = std::make_unique<yggdrasil_decision_forests::dataset::VerticalDataset>(
            std::move(ds));
        ds_ptr = tf_ds.get();

        LOG(INFO) << "Synthetic dataset ready: " << tf_ds->nrow() << " rows, "
                  << data_spec.columns_size() << " columns.";
    } else {
        std::cerr << "Unknown input_mode: " << mode
                  << ". Use 'csv', 'trunk', or 'tfrecord'.\n";
        return 1;
    }

    // 3) Inner sweep: tree_depth x density_factor
    for (int tree_depth : depths_to_sweep) {
    for (float density_factor : densities_to_sweep) {
        LOG(INFO) << "=== Run " << run_idx
                  << ": rows=" << sweep_rows << " cols=" << sweep_cols
                  << " tree_depth=" << tree_depth
                  << " projection_density_factor=" << density_factor << " ===";

    yggdrasil_decision_forests::model::proto::TrainingConfig train_config;
    train_config.set_task(yggdrasil_decision_forests::model::proto::Task::CLASSIFICATION);
    train_config.set_label(label_col);

    yggdrasil_decision_forests::model::proto::DeploymentConfig deploy_config;

    if (num_threads_flag > 0) {
        deploy_config.set_num_threads(num_threads_flag);
    } else if (num_threads_flag == -1) {
        unsigned int cpu_count = std::thread::hardware_concurrency();
        if (cpu_count == 0) cpu_count = 1;
        deploy_config.set_num_threads(cpu_count);
    } else {
        std::cerr << "Invalid value for --num_threads: "
                << num_threads_flag
                << ". Must be >0 for fixed threads or -1 for automatic.\n";
        return 1;
    }

    yggdrasil_decision_forests::model::decision_tree::proto::DecisionTreeTrainingConfig* dt_config;

    if (ensemble_method == "Bagging") {
        train_config.set_learner("RANDOM_FOREST");
        auto& rf = *train_config.MutableExtension(
            yggdrasil_decision_forests::model::random_forest::proto::random_forest_config);
        rf.set_num_trees(num_trees);
        rf.set_bootstrap_training_dataset(absl::GetFlag(FLAGS_bootstrap_training_dataset));
        rf.set_bootstrap_size_ratio(1.0);
        rf.set_winner_take_all_inference(false);
        rf.set_compute_oob_performances(
            absl::GetFlag(FLAGS_compute_oob_performances));
        dt_config = rf.mutable_decision_tree();
    }
    else {
        std::cerr << "Unknown ensemble_method: " << ensemble_method
                << ". Use 'Bagging' or 'Boosting'.\n";
        return 1;
    }

    train_config.set_random_seed(seed);
    dt_config->set_max_depth(tree_depth);
    dt_config->set_min_examples(ensemble_method == "Boosting" ? 5 : 1);

    if (growing_strategy == "GlobalBestFirst") {
        dt_config->mutable_growing_strategy_best_first_global()->set_max_num_nodes(-1);
    } else if (growing_strategy != "Local") {
        std::cerr << "Unknown growing_strategy: " << growing_strategy
                << ". Use Local or GlobalBestFirst.\n";
        return 1;
    }

    if (feature_split_type == "Oblique") {
        auto* sos = dt_config->mutable_sparse_oblique_split();
        sos->set_max_num_projections(absl::GetFlag(FLAGS_max_num_projections));
        sos->set_projection_density_factor(density_factor);
        sos->set_num_projections_exponent(absl::GetFlag(FLAGS_num_projections_exponent));
    } else if (feature_split_type == "Axis Aligned") {
    } else {
        std::cerr << "Unknown feature_split_type: " << feature_split_type
                  << ". Use 'Axis Aligned' or 'Oblique'.\n";
        return 1;
    }

    auto* numerical_split = dt_config->mutable_numerical_split();

    if (hist_type == "Exact") {
        numerical_split->set_type(
            yggdrasil_decision_forests::model::decision_tree::proto::NumericalSplit::EXACT);
    } else if (hist_type == "Random") {
        numerical_split->set_type(
            yggdrasil_decision_forests::model::decision_tree::proto::NumericalSplit::HISTOGRAM_RANDOM);
        numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    } else if (hist_type == "Equal Width") {
        numerical_split->set_type(
            yggdrasil_decision_forests::model::decision_tree::proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH);
        numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    } else {
        std::cerr << "Unknown histogram type: " << hist_type
                << ". Use 'Exact', 'Random', or 'Equal Width'.\n";
        return 1;
    }

    std::unique_ptr<yggdrasil_decision_forests::model::AbstractLearner> learner;
    CHECK_OK(yggdrasil_decision_forests::model::GetLearner(train_config, &learner, deploy_config));

    const int first_tree_id = next_tree_id;
    auto t_train_start = std::chrono::high_resolution_clock::now();

    absl::StatusOr<std::unique_ptr<yggdrasil_decision_forests::model::AbstractModel>> model_or;

    if (ds_ptr != nullptr) {
        model_or = learner->TrainWithStatus(*ds_ptr);
    } else {
        model_or = learner->TrainWithStatus("csv:" + csv_path, data_spec);
    }

    auto t_train_end = std::chrono::high_resolution_clock::now();
    double train_time_s = std::chrono::duration<double>(t_train_end - t_train_start).count();

    if (!model_or.ok()) {
        std::cerr << "Run " << run_idx << " training failed: " << model_or.status().message() << std::endl;
        ++run_idx;
        continue;
    }

    auto model_ptr = std::move(model_or.value());
    next_tree_id += num_trees;

    // Parse per_depth_timing.csv for AP kernel totals from this run's trees
    auto tree_timings = ParsePerDepthTimingCSV("per_depth_timing.csv", first_tree_id);
    double sum_ap_kernel_ms = 0, sum_tree_total_ms = 0;
    double sum_sampling_ms = 0, sum_hist_ms = 0, sum_free_ms = 0;
    double sum_split_ms = 0, sum_node_train_ms = 0;
    double sum_drsw_setup_ms = 0, sum_drsw_proj_ms = 0;
    double sum_compute_offsets_ms = 0;
    double sum_setup_ms = 0, sum_gpu_cleanup_ms = 0, sum_partition_free_ms = 0;
    // Per-tree canonical buckets (same formulas as the run-average below).
    auto tree_buckets = [](const PerTreeTiming& tt, double& projection,
                           double& example_mgmt, double& overhead) {
        projection   = tt.ap_kernel_ms - tt.drsw_setup_ms;  // drsw_setup -> example_mgmt
        if (projection < 0) projection = 0;
        example_mgmt = tt.compute_offsets_ms + tt.drsw_setup_ms + tt.split_ms;
        overhead = tt.tree_total_ms -
            (tt.sampling_ms + projection + tt.hist_ms + example_mgmt + tt.node_train_ms);
        if (overhead < 0) overhead = 0;
    };
    for (const auto& tt : tree_timings) {
        sum_ap_kernel_ms += tt.ap_kernel_ms;
        sum_tree_total_ms += tt.tree_total_ms;
        sum_sampling_ms += tt.sampling_ms;
        sum_hist_ms += tt.hist_ms;
        sum_free_ms += tt.free_ms;
        sum_split_ms += tt.split_ms;
        sum_node_train_ms += tt.node_train_ms;
        sum_drsw_setup_ms += tt.drsw_setup_ms;
        sum_drsw_proj_ms += tt.drsw_proj_ms;
        sum_compute_offsets_ms += tt.compute_offsets_ms;
        sum_setup_ms += tt.setup_ms;
        sum_gpu_cleanup_ms += tt.gpu_cleanup_ms;
        sum_partition_free_ms += tt.partition_free_ms;
        if (per_tree_csv.is_open()) {
            double p_projection, p_example_mgmt, p_overhead;
            tree_buckets(tt, p_projection, p_example_mgmt, p_overhead);
            double p_gap = p_overhead - (tt.setup_ms + tt.gpu_cleanup_ms + tt.partition_free_ms);
            per_tree_csv << run_idx << "," << tt.tree_id << ","
                         << tt.sampling_ms << "," << p_projection << ","
                         << tt.hist_ms << "," << p_example_mgmt << ","
                         << tt.node_train_ms << "," << p_overhead << "," << tt.tree_total_ms << ","
                         << tt.compute_offsets_ms << "," << tt.drsw_setup_ms << "," << tt.split_ms << ","
                         << tt.setup_ms << "," << tt.gpu_cleanup_ms << ","
                         << tt.partition_free_ms << "," << p_gap << ","
                         << tt.ap_kernel_ms << "," << tt.drsw_proj_ms << "," << tt.hist_ms << "\n";
        }
    }
    const int n_trees_parsed = static_cast<int>(tree_timings.size());
    auto avg = [&](double s) { return n_trees_parsed > 0 ? s / n_trees_parsed : 0.0; };
    // Raw per-stage averages (drill-down).
    double ap_kernel_ms = avg(sum_ap_kernel_ms);
    double tree_total_ms = avg(sum_tree_total_ms);
    double sampling_ms = avg(sum_sampling_ms);
    double hist_ms = avg(sum_hist_ms);
    double free_ms = avg(sum_free_ms);
    double split_ms = avg(sum_split_ms);          // partition (SplitUpdate / SplitExamplesAllDepth)
    double node_train_ms = avg(sum_node_train_ms);
    double drsw_setup_ms = avg(sum_drsw_setup_ms);
    double drsw_proj_ms = avg(sum_drsw_proj_ms);
    double compute_offsets_ms = avg(sum_compute_offsets_ms);  // ComputeOffsets (NID only)
    double setup_ms = avg(sum_setup_ms);
    double gpu_cleanup_ms = avg(sum_gpu_cleanup_ms);
    double partition_free_ms = avg(sum_partition_free_ms);

    // --- Canonical buckets: same meaning in every variant, sum == tree_total ---
    //   projection   = apply_proj_kernel bracket MINUS drsw_setup (the DRSW_1
    //                  projection kernel + its prefix-sum setup). drsw_setup is
    //                  reclassified below as example management.
    //   split_finding = minmax + histogram + find-best (eval_proj).
    //   example_mgmt = node<->example structure maintenance, the directly
    //                  comparable cost across variants:
    //                    NID          : compute_offsets (ComputeOffsets) + partition (SplitUpdate)
    //                    DRSW-original : drsw_setup (ComputeDRSWArrays derives node_ids/
    //                                   offsets) + partition (SplitExamplesAllDepth reorder)
    //                    SRDW         : partition only (SplitExamplesAllDepth reorder)
    //   overhead     = residual (setup + gpu cleanup + frees + misc), so the
    //                  buckets close exactly to tree_total.
    double b_sampling      = sampling_ms;
    double b_projection    = ap_kernel_ms - drsw_setup_ms;  // drsw_setup -> example_mgmt
    if (b_projection < 0) b_projection = 0;
    double b_split_finding = hist_ms;
    double b_example_mgmt  = compute_offsets_ms + drsw_setup_ms + split_ms;
    double b_node_train    = node_train_ms;
    double b_overhead      = tree_total_ms -
        (b_sampling + b_projection + b_split_finding + b_example_mgmt + b_node_train);
    if (b_overhead < 0) b_overhead = 0;

    LOG(INFO) << "Run " << run_idx << " results:"
              << " rows=" << sweep_rows << " cols=" << sweep_cols
              << " train_time=" << train_time_s << "s"
              << " trees=" << n_trees_parsed
              << " sampling=" << b_sampling << "ms"
              << " projection=" << b_projection << "ms"
              << " split_finding=" << b_split_finding << "ms"
              << " example_mgmt=" << b_example_mgmt << "ms"
              << " node_train=" << b_node_train << "ms"
              << " overhead=" << b_overhead << "ms"
              << " tree_total=" << tree_total_ms << "ms";
    LOG(INFO) << "Run " << run_idx << " raw:"
              << " compute_offsets=" << compute_offsets_ms << "ms"
              << " partition=" << split_ms << "ms"
              << " apply_proj_kernel=" << ap_kernel_ms << "ms"
              << " drsw_setup=" << drsw_setup_ms << "ms"
              << " drsw_proj=" << drsw_proj_ms << "ms"
              << " eval_proj=" << hist_ms << "ms"
              << " free=" << free_ms << "ms";
    // example_mgmt sub-parts: node<->example structure maintenance.
    LOG(INFO) << "Run " << run_idx << " example_mgmt=" << b_example_mgmt << "ms ="
              << " compute_offsets=" << compute_offsets_ms << "ms"      // NID ComputeOffsets
              << " + drsw_setup=" << drsw_setup_ms << "ms"              // DRSW-original ComputeDRSWArrays
              << " + partition=" << split_ms << "ms";                  // SplitUpdate / SplitExamplesAllDepth
    // overhead sub-parts: per-tree fixed costs (gap = unattributed residual).
    double overhead_gap = b_overhead - (setup_ms + gpu_cleanup_ms + partition_free_ms);
    LOG(INFO) << "Run " << run_idx << " overhead=" << b_overhead << "ms ="
              << " setup=" << setup_ms << "ms"
              << " + gpu_cleanup=" << gpu_cleanup_ms << "ms"
              << " + partition_free=" << partition_free_ms << "ms"
              << " + gap=" << overhead_gap << "ms";

    if (results_csv.is_open()) {
        results_csv << run_idx << "," << sweep_rows << "," << sweep_cols << ","
                    << tree_depth << "," << density_factor << ","
                    << num_trees << "," << train_time_s << ","
                    // canonical buckets
                    << b_sampling << "," << b_projection << ","
                    << b_split_finding << "," << b_example_mgmt << ","
                    << b_node_train << "," << b_overhead << "," << tree_total_ms << ","
                    // raw drill-down
                    << compute_offsets_ms << "," << split_ms << "," << ap_kernel_ms << ","
                    << drsw_setup_ms << "," << drsw_proj_ms << ","
                    << hist_ms << "," << free_ms << std::endl;
    }

    if (!base_out_dir.empty()) {
        std::string out_dir = base_out_dir;
        if (rows_to_sweep.size() * cols_to_sweep.size() *
            depths_to_sweep.size() * densities_to_sweep.size() > 1) {
            out_dir = absl::StrCat(base_out_dir, "/r", sweep_rows,
                                   "_c", sweep_cols,
                                   "_depth", tree_depth,
                                   "_density", density_factor);
        }
        auto save_status = yggdrasil_decision_forests::model::SaveModel(out_dir, *model_ptr);
        if (!save_status.ok()) {
            std::cerr << "Could not save model: " << save_status.message() << std::endl;
        } else {
            std::cout << "Model saved to: " << out_dir << std::endl;
        }
    }

    ++run_idx;
    }  // density_factor
    }  // tree_depth
    }  // sweep_cols
    }  // sweep_rows


    // auto dataset = Utils::flattenCSVColumnMajorWithLabels<float, int>(csv_path); //CSVData struct with flattened data (float), labels(int), num_rows, num_cols

    // const size_t num_proj_base = static_cast<int>(std::sqrt(dataset.num_cols));
    // //std::cout << "Number of rows: " << dataset.num_rows << std::endl;
    // //std::cout << "Number of columns: " << dataset.num_cols << std::endl;
    // //std::cout << "Base number of projections (sqrt of num_cols): " << num_proj_base << std::endl;       

    // std::vector<Utils::ExperimentParams> paramList;
    // // example params for testing (selected_features_count, num_elems_per_thread, num_iters, num_nodes, num_proj, alternate, verbose)    
    // std::vector<int> num_node_values = {64}; // Example values for num_nodes
    // // std::vector<int> num_node_values = {64}; // Example values for num_nodes
    // std::vector<int> num_proj_values = {6}; // Example values for num_proj
    // int num_iters = 3; // Example value for num_iters
    // std::vector<bool> alternate_values = {false}; // Example values for alternate
    // // std::vector<int> num_elems_per_threads = {1, 2, 4, 8, 16, 32, 64, 128, 256}; // Example values for num_elems_per_thread
    // std::vector<int> num_elems_per_threads = {1}; // Example values for num_elems_per_thread
    // std::vector<int> selected_features_counts = {1}; // Example values for selected_features_count
    // // std::vector<int> selected_features_counts = {1, 2, 4, 8, 16}; // Example values for selected_features_count
    // // std::vector<int> reorder_strategies = {0, 1, 2, 3, 4, 5, 6, 7}; // Example values for reorder_strategy
    // std::vector<int> reorder_strategies = {6}; // Example values for reorder_strategy
    // // 0=none  1=greedy  2=hungarian_reorder  3=column_auction  4=iter_hungarian  5=SA  6=symmetric  7=iter_hungarian+node_reorder
    // for (int reorder_strategy : reorder_strategies) {
    //     for (int selected_features_count : selected_features_counts) {
    //         for (int num_elems_per_thread : num_elems_per_threads) {
    //             for (bool alternate : alternate_values) {
    //                 for (int num_nodes : num_node_values) {
    //                     for (int num_proj : num_proj_values) {
    //                         paramList.push_back({selected_features_count, num_elems_per_thread, num_iters, num_nodes, num_proj, alternate, false, reorder_strategy});
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // for (int i = 0; i < paramList.size(); ++i)
    // {
    //     paramList[i].verbose = false; // Set verbose to true for all experiments for detailed output
    //     std::cout << "Experiment " << i
    //             << " (alternate: "               << (paramList[i].alternate ? "true" : "false")
    //             << ", num_nodes: "               << paramList[i].num_nodes
    //             << ", num_proj: "                << paramList[i].num_proj
    //             << ") selected_features_count: " << paramList[i].selected_features_count
    //             << ", num_elems_per_thread: "    << paramList[i].num_elems_per_thread
    //             << ", num_iters: "               << paramList[i].num_iters
    //             << ", reorder: "                 << paramList[i].reorder_strategy
    //             << " (" << mcg::strategy_name(paramList[i].reorder_strategy) << ")"
    //             << ", verbose: "                 << (paramList[i].verbose ? "true" : "false")
    //             << std::endl;
    //     runExperiment(dataset, paramList[i], i); 
    // }
    return 0;
}    

