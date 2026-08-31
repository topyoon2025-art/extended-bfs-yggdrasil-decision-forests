#include "oblique_apply_projection.cuh"
#include "parallel_utils.hpp"
#include "yggdrasil_decision_forests/learner/decision_tree/maximize_groupings.h"
#include <cfloat>

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
#include "absl/strings/str_split.h"

ABSL_FLAG(std::string, input_mode, "csv",
          "Data input mode: csv, Uniform synthetic, Trunk Synthetic, or tfrecord.");
ABSL_FLAG(std::string, train_csv, "",
          "Path to training CSV file.");
ABSL_FLAG(std::string, label_col, "target",
          "Name of label column (used in all modes).");

ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(int, num_trees, 100, "Number of trees in the random forest.");
ABSL_FLAG(int, tree_depth, 16, "Maximum depth of trees (-1 for unlimited).");

ABSL_FLAG(std::string, feature_split_type, "Oblique",
          "Type of feature splits in decision trees: 'Axis Aligned' or 'Oblique'.");

// Oblique split parameters (only used when feature_split_type = "Oblique")
ABSL_FLAG(int, max_num_projections, 64,
          "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, num_projections_exponent, .5,
          "Exponent to determine number of projections.");
ABSL_FLAG(float, projection_density_factor, 2.0f,
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

ABSL_DECLARE_FLAG(bool, combined);

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
    if (csv_path.empty()) {
        std::cerr << "Error: --train_csv is required.\n";
        return 1;
    }

    const auto mode = absl::GetFlag(FLAGS_input_mode);

    // Validate required input_mode flag
    if (mode.empty()) {
        std::cerr << "Error: --input_mode is required. Use csv, synthetic, or tfrecord.\n";
        return 1;
    }

    yggdrasil_decision_forests::dataset::proto::DataSpecification data_spec;
    std::unique_ptr<yggdrasil_decision_forests::dataset::VerticalDataset> tf_ds;
    yggdrasil_decision_forests::dataset::VerticalDataset* ds_ptr = nullptr;
    std::string label_col;

    // 1) Prepare data source based on mode
    if (mode == "csv") {

        label_col = absl::GetFlag(FLAGS_label_col);

        // Validate required CSV parameters
        if (csv_path.empty()) {
        std::cerr << "Error: --train_csv is required in csv mode.\n";
        return 1;
        }
        if (label_col.empty()) {
        std::cerr << "Error: --label_col is required in csv mode.\n";
        return 1;
        }

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
    /////////////////
    // 2) Configure learner
    yggdrasil_decision_forests::model::proto::TrainingConfig train_config;
    train_config.set_task(yggdrasil_decision_forests::model::proto::Task::CLASSIFICATION);
    train_config.set_label(label_col);

    yggdrasil_decision_forests::model::proto::DeploymentConfig deploy_config;
    // GPU vs CPU is chosen at compile time via --config=oblique_gpu; the runtime
    // toggle is commented out until hybrid CPU-GPU offloading is added.
    // deploy_config.set_use_gpu(absl::GetFlag(FLAGS_use_gpu));

    /* #region Handle num_threads */
    int num_threads_flag = absl::GetFlag(FLAGS_num_threads);  // Number of threads to use (1 for single-threaded, -1 for automatic detection)
    if (num_threads_flag > 0) {
        LOG(INFO) << "Running with " << num_threads_flag << " threads, as requested.";
        deploy_config.set_num_threads(num_threads_flag);

    } else if (num_threads_flag == -1) {
        // Automatically detect number of CPUs
        unsigned int cpu_count = std::thread::hardware_concurrency();
        if (cpu_count == 0) {
        cpu_count = 1;  // fallback if detection fails
        }
        LOG(INFO) << "-1 (automatic) threads requested. "
                << cpu_count << " threads set.";
        deploy_config.set_num_threads(cpu_count);

    } else {
        std::cerr << "Invalid value for --num_threads: "
                << num_threads_flag
                << ". Must be >0 for fixed threads or -1 for automatic.\n";
        return 1;
    }
    // Configure ensemble method: Bagging (RF) or Boosting (GBT/MART)
    const std::string ensemble_method = absl::GetFlag(FLAGS_ensemble_method);
    yggdrasil_decision_forests::model::decision_tree::proto::DecisionTreeTrainingConfig* dt_config;

    if (ensemble_method == "Bagging") {
        train_config.set_learner("RANDOM_FOREST");
        auto& rf = *train_config.MutableExtension(
            yggdrasil_decision_forests::model::random_forest::proto::random_forest_config);
        rf.set_num_trees(absl::GetFlag(FLAGS_num_trees));
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

    train_config.set_random_seed(absl::GetFlag(FLAGS_seed));
    int tree_depth = absl::GetFlag(FLAGS_tree_depth);
    dt_config->set_max_depth(tree_depth);
    dt_config->set_min_examples(ensemble_method == "Boosting" ? 5 : 1);

    const auto growing_strategy = absl::GetFlag(FLAGS_growing_strategy);
    if (growing_strategy == "GlobalBestFirst") {
        dt_config->mutable_growing_strategy_best_first_global()->set_max_num_nodes(-1);
    } else if (growing_strategy != "Local") {
        std::cerr << "Unknown growing_strategy: " << growing_strategy
                << ". Use Local or GlobalBestFirst.\n";
        return 1;
    }

    const std::string feature_split_type = absl::GetFlag(FLAGS_feature_split_type);
    if (feature_split_type == "Oblique") {
        LOG(INFO) << "Configuring oblique splits";
        auto* sos = dt_config->mutable_sparse_oblique_split();
        sos->set_max_num_projections(absl::GetFlag(FLAGS_max_num_projections));
        sos->set_projection_density_factor(absl::GetFlag(FLAGS_projection_density_factor));
        sos->set_num_projections_exponent(absl::GetFlag(FLAGS_num_projections_exponent));
    } else if (feature_split_type == "Axis Aligned") {
        LOG(INFO) << "Using axis-aligned splits";
    } else {
        std::cerr << "Unknown feature_split_type: " << feature_split_type
                  << ". Use 'Axis Aligned' or 'Oblique'.\n";
        return 1;
    }

    auto* numerical_split = dt_config->mutable_numerical_split();
  
    const std::string hist_type = absl::GetFlag(FLAGS_numerical_split_type);
    if (hist_type == "Exact") {
        numerical_split->set_type(
            yggdrasil_decision_forests::model::decision_tree::proto::NumericalSplit::EXACT);
        LOG(INFO) << "Using exact splitting";
    } else if (hist_type == "Random") {
        numerical_split->set_type(
            yggdrasil_decision_forests::model::decision_tree::proto::NumericalSplit::HISTOGRAM_RANDOM);
        numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
        LOG(INFO) << "Using random histogram splitting with "
                  << absl::GetFlag(FLAGS_histogram_num_bins) << " bins";
    } else if (hist_type == "Equal Width") {
        numerical_split->set_type(
            yggdrasil_decision_forests::model::decision_tree::proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH);
        numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
        LOG(INFO) << "Using equal width histogram splitting with "
                  << absl::GetFlag(FLAGS_histogram_num_bins) << " bins";
    } else {
        std::cerr << "Unknown histogram type: " << hist_type
                << ". Use 'Exact', 'Random', or 'Equal Width'.\n";
        return 1;
    }

    std::unique_ptr<yggdrasil_decision_forests::model::AbstractLearner> learner;
    CHECK_OK(yggdrasil_decision_forests::model::GetLearner(train_config, &learner, deploy_config));

    absl::StatusOr<std::unique_ptr<yggdrasil_decision_forests::model::AbstractModel>> model_or;

    if (mode == "csv") {
    model_or = learner->TrainWithStatus("csv:" + csv_path, data_spec);
    } else {
        model_or = learner->TrainWithStatus(*ds_ptr);
    }

    if (!model_or.ok()) {
        std::cerr << "Training failed: " << model_or.status().message() << std::endl;
        return 1;
    }

    auto model_ptr = std::move(model_or.value());
    ///////////////////
     // 4) Save model if requested
    const std::string out_dir = absl::GetFlag(FLAGS_model_out_dir);
    if (!out_dir.empty()) {
        auto save_status = yggdrasil_decision_forests::model::SaveModel(out_dir, *model_ptr);
        if (!save_status.ok()) {
        std::cerr << "Could not save model: " << save_status.message() << std::endl;
        return 1;
        }
        std::cout << "Model saved to: " << out_dir << std::endl;
    }


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

