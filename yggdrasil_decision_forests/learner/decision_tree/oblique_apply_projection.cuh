#pragma once

#include <optional>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace random_forest {
namespace proto {
class RandomForestTrainingConfig;
}  // namespace proto
}  // namespace random_forest
}  // namespace model
}  // namespace yggdrasil_decision_forests

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _status = (call);                                         \
        if (_status != cudaSuccess) {                                         \
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(_status)        \
                      << " (code " << _status << ") "                         \
                      << "in " << __FILE__ << ':' << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)


void cuda_warmup();

absl::Status SampleTrainingExamplesKernel_wrap(
    const int num_samples,
    const yggdrasil_decision_forests::model::random_forest::proto::RandomForestTrainingConfig& rf_config,
    std::optional<double> bootstrap_size_ratio_factor,
    int* d_selected_examples,
    std::vector<yggdrasil_decision_forests::model::UnsignedExampleIdx>* selected_examples,
    unsigned long long seed);

absl::Status SampleTrainingExamplesKernel_No_D2H_wrap(
    const int num_rows,
    const yggdrasil_decision_forests::model::random_forest::proto::RandomForestTrainingConfig& rf_config,
    std::optional<double> bootstrap_size_ratio_factor,
    int*& d_selected_examples,
    int& num_samples_out,
    unsigned long long seed);

absl::StatusOr<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer> SplitExamplesInPlaceKernel_wrap(
    const float* d_col_add_projected,
    const float threshold,
    yggdrasil_decision_forests::model::decision_tree::SelectedExamplesRollingBuffer selected_examples,
    int rows_start,
    int rows_node,
    int best_projection_index,
    int num_proj);

absl::StatusOr<std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer>> SplitExamplesInPlaceKernel_all_depth_wrap(
                const float* d_col_add_projected,
                const int* d_selected_examples,
                const int* d_best_proj_per_node,
                const float* d_best_threshold_per_node,
                const int* d_node_row_off,
                const int* node_row_off,
                const int* d_node_block_off,
                const int total_node_blocks,
                std::vector<absl::Span<yggdrasil_decision_forests::model::UnsignedExampleIdx>>& sel_spans,
                std::vector<absl::Span<yggdrasil_decision_forests::model::UnsignedExampleIdx>>& inactive_spans,
                const int num_proj,
                const int num_nodes,
                const int total_rows,
                const int max_rows_per_node,
                const int* d_labels,
                const int num_classes,
                std::vector<std::vector<int>>& pos_class_counts,
                std::vector<std::vector<int>>& neg_class_counts);

absl::StatusOr<std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer>> SplitExamplesInPlaceKernel_all_depth_No_D2H_wrap(
                const float* d_col_add_projected,
                int* d_selected_examples,
                const int* d_best_proj_per_node,
                const float* d_best_threshold_per_node,
                const int* d_node_row_off,
                const int* node_row_off,
                const int* d_node_block_off,
                const int total_node_blocks,
                const int num_proj,
                const int num_nodes,
                const int total_rows,
                const int max_rows_per_node,
                const int* d_labels,
                const int num_classes,
                std::vector<std::vector<int>>& pos_class_counts,
                std::vector<std::vector<int>>& neg_class_counts,
                std::vector<int>& pos_counts,
                std::vector<int>& neg_counts);

absl::StatusOr<std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer>> SplitExamplesInPlaceKernel_all_depth_No_D2H_fused_wrap(
                const float* d_col_add_projected,
                int* d_selected_examples,
                const int* d_best_proj_per_node,
                const float* d_best_threshold_per_node,
                const int* d_node_row_off,
                const int* node_row_off,
                const int* d_node_block_off,
                const int total_node_blocks,
                const int num_proj,
                const int num_nodes,
                const int total_rows,
                const int max_rows_per_node,
                const int* d_labels,
                const int num_classes,
                std::vector<std::vector<int>>& pos_class_counts,
                std::vector<std::vector<int>>& neg_class_counts,
                std::vector<int>& pos_counts,
                std::vector<int>& neg_counts,
                const std::vector<bool>& node_has_split,
                std::vector<int>& child_offsets);

void BlockMinMax_wrap(
    const float* d_col_add_projected,
    const int* d_node_row_off,
    float* d_block_min,
    float* d_block_max,
    const int* d_node_block_off,
    int num_nodes,
    int num_proj,
    int blocks_per_node);

void BlockMinMax_2D_wrap(
    const float* d_col_add_projected,
    const int* d_node_row_off,
    float* d_block_min,
    float* d_block_max,
    const int* d_node_block_off,
    int num_nodes,
    int num_proj,
    int total_blocks);

void finalMinMaxReduceAtomic_wrap(
    float* d_block_min,
    float* d_block_max,
    const int* d_node_block_off,
    float* d_min_vals,
    float* d_max_vals,
    int num_nodes,
    int num_proj,
    int blocks_per_node);

void finalMinMaxReduceAtomic_2D_wrap(
    float* d_block_min,
    float* d_block_max,
    const int* d_node_block_off,
    const int* d_meta_block_off,
    float* d_min_vals,
    float* d_max_vals,
    int num_nodes,
    int num_proj,
    int total_meta_blocks);

void RandomBoundaries_wrap(const int splits_per_segment,
                            float* d_min_val,
                            float* d_max_val,
                            float* d_sorted_candidate_splits,
                            const int num_segments,
                            const int rng_seed);

void BuildHistogramRandomKernel_wrap(const float* d_col_add_projected,
                                    const int* d_selected_examples,
                                    const int* d_labels,
                                    int* d_hist_class0,
                                    int* d_hist_class1,
                                    int num_proj,
                                    int num_bins,
                                    const float* d_sorted_candidate_splits,
                                    const int* d_node_row_off,
                                    int num_nodes,
                                    int blocks_per_node);

void BuildHistogramRandomKernel_2D_wrap(const float* d_col_add_projected,
                                    const int* d_selected_examples,
                                    const int* d_labels,
                                    int* d_hist_class0,
                                    int* d_hist_class1,
                                    int num_proj,
                                    int num_bins,
                                    const float* d_sorted_candidate_splits,
                                    const int* d_node_row_off,
                                    const int* d_node_block_off,
                                    int num_nodes,
                                    int total_node_blocks);

void FindBestEntropySplitKernel_wrap(
    const int* d_hist_class0,
    const int* d_hist_class1,
    float* d_best_gain_per_seg,
    int* d_best_split_per_seg,
    const int num_segments,
    const int num_bins,
    const int num_nodes,
    const int num_proj,
    const int hist_elems,
    int* d_prefix_0,
    int* d_prefix_1,
    float* d_out_per_bin_per_proj);

void BestProjectionPerNodeKernel_wrap(const float* d_best_gain_per_seg,
                            const int* d_best_split_per_seg,
                            float* d_best_gain_per_node,
                            int* d_best_proj_per_node,
                            int* d_best_split_per_node,
                            int* d_num_pos_per_node,
                            float* d_best_threshold_per_node,
                            const int num_nodes,
                            const int num_proj,
                            const int num_bins,
                            const int* d_prefix_0,
                            const int* d_prefix_1,
                            const float* d_candidate_splits);

void ColumnAddProjectionKernel_SRDW_1_2D_perthread_wrap(
    const float* d_flat_data,
    int* d_selected_examples,
    float* d_col_add_projected,
    const int* d_offset,
    const int* d_flat_projection_col_idx,
    const float* d_flat_projection_weights,
    const int* d_node_row_off,
    int num_nodes,
    int num_proj,
    int num_rows,
    int total_rows);

void ColumnAddProjectionKernel_SRDW_1_wrap(
    const float* d_flat_data,
    int* d_selected_examples,
    float* d_col_add_projected,
    const int* d_offset,
    const int* d_flat_projection_col_idx,
    const float* d_flat_projection_weights,
    const int* d_node_row_off,
    int num_nodes,
    int num_proj,
    int num_rows,
    int total_rows,
    int blocks_per_node);

void ColumnAddProjectionKernel_SRDW_1_2D_wrap(
    const float* d_flat_data,
    int* d_selected_examples,
    float* d_col_add_projected,
    const int* d_offset,
    const int* d_flat_projection_col_idx,
    const float* d_flat_projection_weights,
    const int* d_node_row_off,
    const int* d_node_block_off,
    int num_nodes,
    int num_proj,
    int num_rows,
    int total_rows,
    int total_blocks);

void ColumnAddProjectionKernel_DRSW_1_wrap(
    const float* d_flat_data,
    const int* d_selected_examples,
    float* d_col_add_projected,
    int* d_node_ids,
    int* d_node_offsets,
    const int* d_offset,
    const int* d_flat_projection_col_idx,
    const float* d_flat_projection_weights,
    const int* d_node_row_off,
    int num_proj,
    int num_rows,
    int num_nodes,
    int num_total_rows);

void ApplyProjectionBaseline (const float* d_flat_data,
                              int* d_selected_examples,//selected examples indices
                              float* d_col_add_projected,
                              int* d_node_ids,
                              int* d_node_offsets,
                              int* d_node_counts,
                              int* d_node_row_start_by_row,
                              const int num_proj,
                              const int num_total_rows,
                              const int num_segments,
                              const int num_nodes,
                              const int num_elems_per_thread,
                              std::vector<int>& node_start_idx,
                              std::vector<int>& node_count,
                              int selected_features_count,
                              bool alternate,
                              float* d_min_vals,
                              float* d_max_vals,
                              int max_rows_per_node,
                              int* node_row_off,
                              int* d_node_row_off,
                              int* d_offset,
                              int* d_flat_projection_col_idx,
                              float* d_flat_projection_weights,
                              int* d_flat_projection_col_idx_shared,
                              float* d_flat_projection_weights_shared,
                              bool verbose);

void ApplyProjectionColumnADD (const float* d_flat_data,
                                unsigned int* d_selected_examples,//selected examples indices
                                float* d_col_add_projected,
                                float** d_min_vals_out,
                                float** d_max_vals_out,
                                float** d_bin_widths_out,
                                std::vector<std::vector<std::vector<int>>>& projection_col_idx,
                                std::vector<std::vector<std::vector<float>>>& projection_weights,
                                const int num_rows,  //num_rows
                                const int num_proj, //num_proj
                                const unsigned int train_dataset,
                                double* elapsed_ms,
                                const int gpu_mode, //0: Exact, 1: Random, 2: Equal Width
                                const bool verbose,
                                std::vector<int>& node_start_idx,
                                std::vector<int>& node_count,
                                const bool separate,
                                const bool cub,
                                const bool fused,
                                int num_elems_per_thread
                              );

void RandomHistogram (const float* __restrict__ d_col_add_projected, //attributes
const int* __restrict__ selected_examples, //selected examples
const int* __restrict__ d_labels,
float* h_min_vals,
float* h_max_vals,
int** d_prefix_0,
int** d_prefix_1,
int** d_prefix_2,
float** d_candidate_splits,
const int num_rows, //selected_examples.size()
const int num_bins,
const int num_proj,
const int num_nodes,
std::vector<int>& node_start_idx, // *** NEW
std::vector<int>& node_count,
std::mt19937& random,
const bool verbose
);

void ThrustSortIndicesOnly(float* d_proj_values, 
                          unsigned int* d_row_ids,
                          unsigned int* d_selected_examples, 
                          int num_rows, 
                          int num_proj,
                          std::vector<int>& node_start_idx,
                          std::vector<int>& node_count,
                          bool verbose
                          );

void ExactSplit(
    unsigned int* d_sorted_indices,  // [num_proj * num_rows]
    const unsigned int* d_labels,          // [num_proj * num_rows]
    float* best_gain_out, // [num_proj], initial best gain values
    int* best_split_out, // [num_proj], initial best split values
    float* best_threshold_out,
    int* best_proj,
    const int num_rows,
    const int num_proj,
    float* d_col_add_projected,  // [num_proj * num_rows]
    double* elapsed_ms,
    bool verbose,
    const int comp_method,
    unsigned int* d_selected_examples,
    std::vector<int>& node_start_idx,
    std::vector<int>& node_count
    );

void ApplyProjectionColumnADDFused (const float* d_flat_data,
                                    unsigned int* d_selected_examples,//selected examples indices
                                    float* d_col_add_projected,
                                    float** d_min_vals_out,
                                    float** d_max_vals_out,
                                    float** d_bin_widths_out,
                                    std::vector<std::vector<std::vector<int>>>& projection_col_idx,
                                    std::vector<std::vector<std::vector<float>>>& projection_weights,
                                    const int num_selected_examples,  //num_rows
                                    const int num_proj, //num_proj
                                    const unsigned int num_total_rows,
                                    double* elapsed_apply_ms,
                                    const int gpu_mode, //0: Exact, 1: Random, 2: Equal Width
                                    const bool verbose,
                                    std::vector<int>& node_start_idx,
                                    std::vector<int>& node_count,
                                    const bool separate,
                                    const bool cub,
                                    const bool fused,
                                    const int num_bins,
                                    const int num_nodes,
                                    const unsigned int* __restrict__ d_labels,
                                    float*  h_min_vals,
                                    float*  h_max_vals,
                                    int**   d_prefix_0_out,
                                    int**   d_prefix_1_out,
                                    int**   d_prefix_2_out,
                                    float** d_candidate_splits_out,
                                    std::mt19937& random,
                                    const int col_gen_seed,
                                    const bool one_kernel,
                                    int num_elems_per_thread
                                    );

void PrintAndResetSplitAllDepthTimers(int depth = -1);
