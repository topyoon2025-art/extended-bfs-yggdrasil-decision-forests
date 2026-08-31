#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/thread/thread_search.cuh>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cfloat>
#include <cstring>
#include <iostream>
#include <chrono>
#include <vector>

static double s_sad_malloc = 0, s_sad_flags = 0, s_sad_prefix = 0,
              s_sad_scatter = 0, s_sad_d2h_scatter = 0, s_sad_d2h_counts = 0,
              s_sad_free = 0;

#include "oblique_apply_projection.cuh"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"

__device__ __forceinline__
float entropy(const int pos, const int neg) {
    int total = pos + neg;
    if (total <= 0) return 0.0f;

    float p_pos = float(pos) / float(total);
    float p_neg = float(neg) / float(total);

    const float eps = 1e-30f;   // protects against denormals and log(0)

    float e = 0.0f;
    if (p_pos > eps) e -= p_pos * logf(p_pos);
    if (p_neg > eps) e -= p_neg * logf(p_neg);
    return e;
}

struct GainBin {
    float gain;
    int bin;
};

struct MaxGainBinOp {
    __device__ __forceinline__
    GainBin operator()(const GainBin& a, const GainBin& b) const {
        if (b.gain > a.gain) return b;
        if (b.gain == a.gain && b.bin < a.bin) return b;
        return a;
    }
};

struct GainProjBin {
    float gain;
    int proj;
    int bin;
};

struct MaxGainProjBinOp {
    __device__ __forceinline__
    GainProjBin operator()(const GainProjBin& a, const GainProjBin& b) const {
        if (b.gain > a.gain) return b;
        if (b.gain == a.gain && b.proj < a.proj) return b; // optional tie-break
        return a;
    }
};


__device__ __forceinline__
void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__
void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

struct MinMax {
    float minv;
    float maxv;
};

struct MinMaxOp {
    __device__ __forceinline__
    MinMax operator()(const MinMax& a, const MinMax& b) const {
        return {fminf(a.minv, b.minv), fmaxf(a.maxv, b.maxv)};
    }
};

__global__  void warmup_kernel() {
    // This kernel does nothing but can be used to warm up the GPU
    // to avoid including initialization time in our benchmarks.
}

void cuda_warmup() {
    warmup_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

__global__ void SampleTrainingExamplesKernel(
    int* __restrict__ d_selected_examples,
    const int num_samples,
    const int num_rows,
    const unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;

    curandState state;
    curand_init(seed, tid, 0, &state);

    // Rejection sampling to avoid modulo bias
    const unsigned int limit = (0xFFFFFFFFu / num_rows) * num_rows;
    unsigned int r;
    do { r = curand(&state); } while (r >= limit);
    d_selected_examples[tid] = r % num_rows;
}

absl::Status SampleTrainingExamplesKernel_wrap(
    const int num_rows,
    const yggdrasil_decision_forests::model::random_forest::proto::RandomForestTrainingConfig& rf_config,
    std::optional<double> bootstrap_size_ratio_factor,
    int* d_selected_examples,
    std::vector<yggdrasil_decision_forests::dataset::UnsignedExampleIdx>* selected_examples)
{
    const int num_samples = std::max(
        int64_t{1},
        static_cast<int64_t>(static_cast<double>(num_rows) * rf_config.bootstrap_size_ratio()));

    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_samples * sizeof(int)));

    constexpr int BLOCK = 256;
    const int blocks = (num_samples + BLOCK - 1) / BLOCK;

    unsigned long long seed = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();

    SampleTrainingExamplesKernel<<<blocks, BLOCK>>>(
        d_selected_examples, num_samples, num_rows, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> d_ptr(d_selected_examples);
    thrust::sort(d_ptr, d_ptr + num_samples);

    selected_examples->resize(num_samples);
    CUDA_CHECK(cudaMemcpy(selected_examples->data(), d_selected_examples,
        num_samples * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_selected_examples));
    return absl::OkStatus();
}

__global__ void ComputeSplitFlagsKernel(
    const float* __restrict__ d_col_add_projected,
    const float threshold,
    int* __restrict__ d_flags,
    int rows_start,
    int best_projection_index,
    int rows_node,
    int num_proj)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows_node) return;

    const int base = rows_start * num_proj + best_projection_index * rows_node;
    float val = d_col_add_projected[base + r];
    d_flags[r] = (val >= threshold) ? 1 : 0;
}

__global__ void ScatterSplitKernel(
    const int* __restrict__ d_selected_examples,
    const int* __restrict__ d_flags,
    const int* __restrict__ d_prefix_sum,
    int* __restrict__ d_pos_examples,
    int* __restrict__ d_neg_examples,
    int rows_node)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows_node) return;

    int example_idx = d_selected_examples[r];
    int inclusive_sum = d_prefix_sum[r];

    if (d_flags[r]) {
        d_pos_examples[inclusive_sum - 1] = example_idx;
    } else {
        d_neg_examples[r - inclusive_sum] = example_idx;
    }
}

absl::StatusOr<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer> SplitExamplesInPlaceKernel_wrap(
    const float* d_col_add_projected,
    const float threshold,
    yggdrasil_decision_forests::model::decision_tree::SelectedExamplesRollingBuffer selected_examples,
    int rows_start,
    int rows_node,
    int best_projection_index,
    int num_proj)
{
    using UnsignedExampleIdx = yggdrasil_decision_forests::dataset::UnsignedExampleIdx;
    yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer example_split;

    const int n = selected_examples.active.size();
    constexpr int BLOCK = 256;
    const int num_blocks = (n + BLOCK - 1) / BLOCK;

    int* d_selected_examples;
    int* d_flags;
    int* d_prefix_sum;
    int* d_pos_examples;
    int* d_neg_examples;

    CUDA_CHECK(cudaMalloc(&d_selected_examples, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos_examples, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neg_examples, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_selected_examples, selected_examples.active.data(),
        n * sizeof(int), cudaMemcpyHostToDevice));

    ComputeSplitFlagsKernel<<<num_blocks, BLOCK>>>(
        d_col_add_projected, threshold, d_flags,
        rows_start, best_projection_index, n, num_proj);
    CUDA_CHECK(cudaGetLastError());

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_flags, d_prefix_sum, n);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_flags, d_prefix_sum, n);
    CUDA_CHECK(cudaGetLastError());

    int total_pos;
    CUDA_CHECK(cudaMemcpy(&total_pos, d_prefix_sum + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    int total_neg = n - total_pos;

    ScatterSplitKernel<<<num_blocks, BLOCK>>>(
        d_selected_examples, d_flags, d_prefix_sum,
        d_pos_examples, d_neg_examples, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Copy results into selected_examples.inactive (caller-owned buffer).
    // Positives go to front, negatives go after — matching CPU EvalConditionTemplate.
    if (total_pos > 0) {
        CUDA_CHECK(cudaMemcpy(selected_examples.inactive.data(), d_pos_examples,
            total_pos * sizeof(UnsignedExampleIdx), cudaMemcpyDeviceToHost));
    }
    if (total_neg > 0) {
        CUDA_CHECK(cudaMemcpy(selected_examples.inactive.data() + total_pos, d_neg_examples,
            total_neg * sizeof(UnsignedExampleIdx), cudaMemcpyDeviceToHost));
    }

    // Build spans the same way as CPU EvalConditionTemplate (decision_tree.cc:953-958).
    // .active points into inactive buffer (where we just wrote), .inactive points into
    // active buffer (old data, now available as scratch for next level).
    example_split.positive_examples = {
        .active = selected_examples.inactive.subspan(0, total_pos),
        .inactive = selected_examples.active.subspan(0, total_pos)};
    example_split.negative_examples = {
        .active = selected_examples.inactive.subspan(total_pos),
        .inactive = selected_examples.active.subspan(total_pos)};

    CUDA_CHECK(cudaFree(d_selected_examples));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_prefix_sum));
    CUDA_CHECK(cudaFree(d_pos_examples));
    CUDA_CHECK(cudaFree(d_neg_examples));
    CUDA_CHECK(cudaFree(d_temp));

    return example_split;
}
__global__ void ComputeSplitFlagsAllDepthKernel(
    const float* __restrict__ d_col_add_projected,
    const int* __restrict__ d_best_proj_per_node,
    const float* __restrict__ d_best_threshold_per_node,
    const int* __restrict__ d_node_row_off,
    int* __restrict__ d_flags,
    int num_proj
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int node_id = blockIdx.y;
    float threshold = d_best_threshold_per_node[node_id];
    int rows_start = d_node_row_off[node_id];
    int rows_end = d_node_row_off[node_id + 1];
    int rows_node = rows_end - rows_start;
    if (r >= rows_node) return;
    int best_projection_index = d_best_proj_per_node[node_id];
    if (best_projection_index < 0) {
        d_flags[rows_start + r] = 0;
        return;
    }

    const int base = rows_start * num_proj + best_projection_index * rows_node;

    float val = d_col_add_projected[base + r];
    d_flags[rows_start + r] = val >= threshold ? 1 : 0;
}

__global__ void GatherLastPrefixSumKernel(
    const int* __restrict__ d_prefix_sum,
    const int* __restrict__ d_node_row_off,
    int* __restrict__ d_total_pos,
    int num_nodes)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= num_nodes) return;
    int last_idx = d_node_row_off[n + 1] - 1;
    d_total_pos[n] = d_prefix_sum[last_idx];
}

__global__ void ScatterSplitAllDepthKernel(
    const int* __restrict__ d_selected_examples,
    const int* __restrict__ d_flags,
    const int* __restrict__ d_prefix_sum,
    const int* __restrict__ d_node_row_off,
    int* __restrict__ d_pos_examples,
    int* __restrict__ d_neg_examples,
    const int* __restrict__ d_labels,
    int* __restrict__ d_class_counts,
    const int num_classes)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int node_id = blockIdx.y;
    int rows_start = d_node_row_off[node_id];
    int rows_end = d_node_row_off[node_id + 1];
    int rows_node = rows_end - rows_start;

    if (r >= rows_node) return;

    int example_idx = d_selected_examples[rows_start + r];
    int inclusive_sum = d_prefix_sum[rows_start + r];
    int side = d_flags[rows_start + r];

    if (side) {
        d_pos_examples[rows_start + inclusive_sum - 1] = example_idx;
    } else {
        d_neg_examples[rows_start + r - inclusive_sum] = example_idx;
    }

    int cls = d_labels[example_idx];
    atomicAdd(&d_class_counts[node_id * 2 * num_classes + side * num_classes + cls], 1);
}

// --- PREVIOUS VERSION: exclusive prefix sum + per-node D2H loop ---
// absl::StatusOr<std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer>> SplitExamplesInPlaceKernel_all_depth_wrap(
//                 const float* d_col_add_projected,
//                 const int* d_selected_examples,
//                 const int* d_best_proj_per_node,
//                 const float* d_best_threshold_per_node,
//                 const int* d_node_row_off,
//                 const int* node_row_off,
//                 std::vector<absl::Span<yggdrasil_decision_forests::model::UnsignedExampleIdx>>& sel_spans,
//                 std::vector<absl::Span<yggdrasil_decision_forests::model::UnsignedExampleIdx>>& inactive_spans,
//                 const int num_proj,
//                 const int num_nodes,
//                 const int total_rows,
//                 const int max_rows_per_node,
//                 const int* d_labels,
//                 const int num_classes,
//                 std::vector<std::vector<int>>& pos_class_counts,
//                 std::vector<std::vector<int>>& neg_class_counts)
// {
//     using UnsignedExampleIdx = yggdrasil_decision_forests::dataset::UnsignedExampleIdx;
//     std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer> example_splits(num_nodes);
//
//     constexpr int BLOCK = 256;
//     int blocks = (max_rows_per_node + BLOCK - 1) / BLOCK;
//     dim3 grid(blocks, num_nodes, 1);
//
//     int* d_flags;
//     int* d_prefix_sum;
//     CUDA_CHECK(cudaMalloc(&d_flags, total_rows * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_prefix_sum, total_rows * sizeof(int)));
//     ComputeSplitFlagsAllDepthKernel<<<grid, BLOCK>>>(
//         d_col_add_projected,
//         d_best_proj_per_node,
//         d_best_threshold_per_node,
//         d_node_row_off,
//         d_flags,
//         num_proj
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());
//     CUDA_CHECK(cudaGetLastError());
//
//     // Per-node exclusive prefix sum using cub::DeviceScan
//     void* d_temp = nullptr;
//     size_t temp_bytes = 0;
//     cub::DeviceScan::ExclusiveSum(
//         d_temp, temp_bytes,
//         d_flags, d_prefix_sum,
//         max_rows_per_node);
//     cudaMalloc(&d_temp, temp_bytes);
//
//     int offset = 0;
//     for (int n = 0; n < num_nodes; ++n) {
//       const int rows_n = static_cast<int>(sel_spans[n].size());
//       cub::DeviceScan::ExclusiveSum(
//           d_temp, temp_bytes,
//           d_flags + offset, d_prefix_sum + offset,
//           rows_n);
//       offset += rows_n;
//     }
//     cudaFree(d_temp);
//
//     int* d_pos_examples;
//     int* d_neg_examples;
//     cudaMalloc(&d_pos_examples, total_rows * sizeof(int));
//     cudaMalloc(&d_neg_examples, total_rows * sizeof(int));
//
//     const int counts_size = num_nodes * 2 * num_classes;
//     int* d_class_counts;
//     CUDA_CHECK(cudaMalloc(&d_class_counts, counts_size * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_class_counts, 0, counts_size * sizeof(int)));
//
//     ScatterSplitAllDepthKernel<<<grid, BLOCK>>>(
//         d_selected_examples,
//         d_flags,
//         d_prefix_sum,
//         d_node_row_off,
//         d_pos_examples,
//         d_neg_examples,
//         d_labels,
//         d_class_counts,
//         num_classes);
//     CUDA_CHECK(cudaDeviceSynchronize());
//     CUDA_CHECK(cudaGetLastError());
//
//     std::vector<int> last_flag_per_node(num_nodes);
//     std::vector<int> last_prefix_per_node(num_nodes);
//     std::vector<int> total_pos_per_node(num_nodes);
//     std::vector<int> total_neg_per_node(num_nodes);
//     for (int n = 0; n < num_nodes; ++n) {
//         int rows_node = static_cast<int>(sel_spans[n].size());
//         CUDA_CHECK(cudaMemcpy(&last_flag_per_node[n], d_flags + node_row_off[n] + rows_node - 1, sizeof(int), cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaMemcpy(&last_prefix_per_node[n], d_prefix_sum + node_row_off[n] + rows_node - 1, sizeof(int), cudaMemcpyDeviceToHost));
//         total_pos_per_node[n] = last_prefix_per_node[n] + last_flag_per_node[n];
//         total_neg_per_node[n] = rows_node - total_pos_per_node[n];
//         if (total_pos_per_node[n] > 0) {
//             CUDA_CHECK(cudaMemcpy(inactive_spans[n].data(), d_pos_examples + node_row_off[n], total_pos_per_node[n] * sizeof(UnsignedExampleIdx), cudaMemcpyDeviceToHost));
//         }
//         if (total_neg_per_node[n] > 0) {
//             CUDA_CHECK(cudaMemcpy(inactive_spans[n].data() + total_pos_per_node[n], d_neg_examples + node_row_off[n], total_neg_per_node[n] * sizeof(UnsignedExampleIdx), cudaMemcpyDeviceToHost));
//         }
//         example_splits[n].positive_examples = {
//             .active = inactive_spans[n].subspan(0, total_pos_per_node[n]),
//             .inactive = sel_spans[n].subspan(0, total_pos_per_node[n])};
//         example_splits[n].negative_examples = {
//             .active = inactive_spans[n].subspan(total_pos_per_node[n]),
//             .inactive = sel_spans[n].subspan(total_pos_per_node[n])};
//     }
//
//     std::vector<int> h_class_counts(counts_size);
//     CUDA_CHECK(cudaMemcpy(h_class_counts.data(), d_class_counts,
//                           counts_size * sizeof(int), cudaMemcpyDeviceToHost));
//
//     pos_class_counts.resize(num_nodes);
//     neg_class_counts.resize(num_nodes);
//     for (int n = 0; n < num_nodes; ++n) {
//       pos_class_counts[n].assign(
//           h_class_counts.begin() + n * 2 * num_classes + 1 * num_classes,
//           h_class_counts.begin() + n * 2 * num_classes + 2 * num_classes);
//       neg_class_counts[n].assign(
//           h_class_counts.begin() + n * 2 * num_classes + 0 * num_classes,
//           h_class_counts.begin() + n * 2 * num_classes + 1 * num_classes);
//     }
//
//     cudaFree(d_pos_examples);
//     cudaFree(d_neg_examples);
//     cudaFree(d_flags);
//     cudaFree(d_prefix_sum);
//     cudaFree(d_class_counts);
//
//     return example_splits;
// }

// --- CURRENT VERSION: inclusive prefix sum + bulk D2H + gather kernel ---
absl::StatusOr<std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer>> SplitExamplesInPlaceKernel_all_depth_wrap(
                const float* d_col_add_projected,
                const int* d_selected_examples,
                const int* d_best_proj_per_node,
                const float* d_best_threshold_per_node,
                const int* d_node_row_off,
                const int* node_row_off,
                std::vector<absl::Span<yggdrasil_decision_forests::model::UnsignedExampleIdx>>& sel_spans,
                std::vector<absl::Span<yggdrasil_decision_forests::model::UnsignedExampleIdx>>& inactive_spans,
                const int num_proj,
                const int num_nodes,
                const int total_rows,
                const int max_rows_per_node,
                const int* d_labels,
                const int num_classes,
                std::vector<std::vector<int>>& pos_class_counts,
                std::vector<std::vector<int>>& neg_class_counts)
{
    using UnsignedExampleIdx = yggdrasil_decision_forests::dataset::UnsignedExampleIdx;
    std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer> example_splits(num_nodes);

    auto t0 = std::chrono::steady_clock::now();

    constexpr int BLOCK = 256;
    int blocks = (max_rows_per_node + BLOCK - 1) / BLOCK;
    dim3 grid(blocks, num_nodes, 1);

    int* d_flags;
    int* d_prefix_sum;
    CUDA_CHECK(cudaMalloc(&d_flags, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, total_rows * sizeof(int)));

    auto t1 = std::chrono::steady_clock::now();

    ComputeSplitFlagsAllDepthKernel<<<grid, BLOCK>>>(
        d_col_add_projected,
        d_best_proj_per_node,
        d_best_threshold_per_node,
        d_node_row_off,
        d_flags,
        num_proj
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    auto t2 = std::chrono::steady_clock::now();

    // Per-node inclusive prefix sum using cub::DeviceScan
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        d_flags, d_prefix_sum,
        max_rows_per_node);
    cudaMalloc(&d_temp, temp_bytes);

    int offset = 0;
    for (int n = 0; n < num_nodes; ++n) {
      const int rows_n = static_cast<int>(sel_spans[n].size());
      cub::DeviceScan::InclusiveSum(
          d_temp, temp_bytes,
          d_flags + offset, d_prefix_sum + offset,
          rows_n);
      offset += rows_n;
    }
    cudaFree(d_temp);
    // Now I have d_prefix_sum segmented per node (inclusive)

    auto t3 = std::chrono::steady_clock::now();

    int* d_pos_examples;
    int* d_neg_examples;
    cudaMalloc(&d_pos_examples, total_rows * sizeof(int));
    cudaMalloc(&d_neg_examples, total_rows * sizeof(int));

    const int counts_size = num_nodes * 2 * num_classes;
    int* d_class_counts;
    CUDA_CHECK(cudaMalloc(&d_class_counts, counts_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_class_counts, 0, counts_size * sizeof(int)));

    ScatterSplitAllDepthKernel<<<grid, BLOCK>>>(
        d_selected_examples,
        d_flags,  // flattened for all active rows but per node
        d_prefix_sum, // flattened but per node inclusive prefix sum
        d_node_row_off,
        d_pos_examples,
        d_neg_examples,
        d_labels,
        d_class_counts, // flattened but per node counts for 2 sides
        num_classes);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    auto t4 = std::chrono::steady_clock::now();

    // Gather total_pos per node on device, then 3 bulk D2H copies
    int* d_total_pos;
    CUDA_CHECK(cudaMalloc(&d_total_pos, num_nodes * sizeof(int)));
    {
      int gather_blocks = (num_nodes + 255) / 256;
      GatherLastPrefixSumKernel<<<gather_blocks, 256>>>(
          d_prefix_sum, d_node_row_off, d_total_pos, num_nodes);
    }

    std::vector<int> total_pos_per_node(num_nodes);
    CUDA_CHECK(cudaMemcpy(total_pos_per_node.data(), d_total_pos,
                          num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_total_pos);

    std::vector<int> h_pos_flat(total_rows); // flattened for all active rows but per node for positive examples
    std::vector<int> h_neg_flat(total_rows); // flattened for all active rows but per node for negative examples
    CUDA_CHECK(cudaMemcpy(h_pos_flat.data(), d_pos_examples,
                          total_rows * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_neg_flat.data(), d_neg_examples,
                          total_rows * sizeof(int), cudaMemcpyDeviceToHost));

    for (int n = 0; n < num_nodes; ++n) {
        int rows_node = static_cast<int>(sel_spans[n].size());
        int total_pos = total_pos_per_node[n];
        int total_neg = rows_node - total_pos;
        int off = node_row_off[n];
        if (total_pos > 0) {
            std::memcpy(inactive_spans[n].data(),
                        h_pos_flat.data() + off,
                        total_pos * sizeof(UnsignedExampleIdx));
        }
        if (total_neg > 0) {
            std::memcpy(inactive_spans[n].data() + total_pos,
                        h_neg_flat.data() + off,
                        total_neg * sizeof(UnsignedExampleIdx));
        }
        example_splits[n].positive_examples = {
            .active = inactive_spans[n].subspan(0, total_pos),
            .inactive = sel_spans[n].subspan(0, total_pos)};
        example_splits[n].negative_examples = {
            .active = inactive_spans[n].subspan(total_pos),
            .inactive = sel_spans[n].subspan(total_pos)};
    }

    auto t5 = std::chrono::steady_clock::now();

    std::vector<int> h_class_counts(counts_size);
    CUDA_CHECK(cudaMemcpy(h_class_counts.data(), d_class_counts,
                          counts_size * sizeof(int), cudaMemcpyDeviceToHost));

    pos_class_counts.resize(num_nodes);
    neg_class_counts.resize(num_nodes);
    for (int n = 0; n < num_nodes; ++n) {
      pos_class_counts[n].assign(
          h_class_counts.begin() + n * 2 * num_classes + 1 * num_classes,
          h_class_counts.begin() + n * 2 * num_classes + 2 * num_classes);
      neg_class_counts[n].assign(
          h_class_counts.begin() + n * 2 * num_classes + 0 * num_classes,
          h_class_counts.begin() + n * 2 * num_classes + 1 * num_classes);
    }

    auto t6 = std::chrono::steady_clock::now();

    cudaFree(d_pos_examples);
    cudaFree(d_neg_examples);
    cudaFree(d_flags);
    cudaFree(d_prefix_sum);
    cudaFree(d_class_counts);

    auto t7 = std::chrono::steady_clock::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b - a).count(); };
    std::cerr << "  SplitAllDepth microtiming:"
              << " malloc=" << ms(t0, t1)
              << " flags_kernel=" << ms(t1, t2)
              << " prefix_sum=" << ms(t2, t3)
              << " scatter_kernel=" << ms(t3, t4)
              << " d2h_scatter=" << ms(t4, t5)
              << " d2h_counts=" << ms(t5, t6)
              << " free=" << ms(t6, t7)
              << " TOTAL=" << ms(t0, t7)
              << " nodes=" << num_nodes
              << " total_rows=" << total_rows
              << " max_rows=" << max_rows_per_node
              << "\n";

    s_sad_malloc      += ms(t0, t1);
    s_sad_flags       += ms(t1, t2);
    s_sad_prefix      += ms(t2, t3);
    s_sad_scatter     += ms(t3, t4);
    s_sad_d2h_scatter += ms(t4, t5);
    s_sad_d2h_counts  += ms(t5, t6);
    s_sad_free        += ms(t6, t7);

    return example_splits;
}

void PrintAndResetSplitAllDepthTimers() {
    double total = s_sad_malloc + s_sad_flags + s_sad_prefix + s_sad_scatter
                 + s_sad_d2h_scatter + s_sad_d2h_counts + s_sad_free;
    std::cerr << "SplitAllDepth TOTAL (tree):"
              << " malloc=" << s_sad_malloc
              << " flags_kernel=" << s_sad_flags
              << " prefix_sum=" << s_sad_prefix
              << " scatter_kernel=" << s_sad_scatter
              << " d2h_scatter=" << s_sad_d2h_scatter
              << " d2h_counts=" << s_sad_d2h_counts
              << " free=" << s_sad_free
              << " SUM=" << total << "ms\n";
    s_sad_malloc = s_sad_flags = s_sad_prefix = s_sad_scatter = 0;
    s_sad_d2h_scatter = s_sad_d2h_counts = s_sad_free = 0;
}

template<int BLOCK>
__global__ void BlockMinMax(const float*  __restrict__ projected,
                            int           num_proj,
                            const int*    __restrict__ node_row_off,
                            const int*    __restrict__ node_block_off,
                            float*        __restrict__ d_block_min,
                            float*        __restrict__ d_block_max)
{
    int node_id = blockIdx.y;
    int proj_id = blockIdx.z;

    int row_start = node_row_off[node_id];
    int rows_node = node_row_off[node_id + 1] - row_start;

    if (rows_node == 0) return;

    int blocks_node = node_block_off[node_id + 1] - node_block_off[node_id];
    if (blockIdx.x >= blocks_node) return;

    int r_local = blockIdx.x * blockDim.x + threadIdx.x;

    float local_min =  FLT_MAX;
    float local_max = -FLT_MAX;

    if (r_local < rows_node) {
        const int base = row_start * num_proj + proj_id * rows_node;
        float v = projected[base + r_local];

        local_min = v;
        local_max = v;
    }

    using BlockReduce = cub::BlockReduce<MinMax, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    MinMax val;
    val.minv = local_min;
    val.maxv = local_max;

    MinMax block_mm = BlockReduce(temp_storage).Reduce(val, MinMaxOp{});

    if (threadIdx.x == 0) {
        int block_id = num_proj * node_block_off[node_id] + proj_id * blocks_node + blockIdx.x;
        d_block_min[block_id] = block_mm.minv;
        d_block_max[block_id] = block_mm.maxv;
    }
}

void BlockMinMax_wrap(
    const float* d_col_add_projected,
    const int* d_node_row_off,
    float* d_block_min,
    float* d_block_max,
    const int* d_node_block_off,
    int num_nodes,
    int num_proj,
    int blocks_per_node)
{
    constexpr int BLOCK = 256;

    dim3 grid(blocks_per_node, num_nodes, num_proj);
    BlockMinMax<BLOCK><<<grid, BLOCK>>>(d_col_add_projected,
                                        num_proj,
                                        d_node_row_off,
                                        d_node_block_off,
                                        d_block_min,
                                        d_block_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK>
__global__ void BlockMinMax_2D(const float*  __restrict__ projected,
                               int           num_proj,
                               const int*    __restrict__ node_row_off,
                               const int*    __restrict__ node_block_off,
                               float*        __restrict__ d_block_min,
                               float*        __restrict__ d_block_max,
                               int           num_nodes)
{
    const int proj_id = blockIdx.y;

    const int node_id = cub::UpperBound(node_block_off, num_nodes + 1, (int)blockIdx.x) - 1;
    const int local_bx = blockIdx.x - node_block_off[node_id];
    const int blocks_node = node_block_off[node_id + 1] - node_block_off[node_id];

    int row_start = node_row_off[node_id];
    int rows_node = node_row_off[node_id + 1] - row_start;

    int r_local = local_bx * blockDim.x + threadIdx.x;

    float local_min =  FLT_MAX;
    float local_max = -FLT_MAX;

    if (r_local < rows_node) {
        const int base = row_start * num_proj + proj_id * rows_node;
        float v = projected[base + r_local];
        local_min = v;
        local_max = v;
    }

    using BlockReduce = cub::BlockReduce<MinMax, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    MinMax val;
    val.minv = local_min;
    val.maxv = local_max;

    MinMax block_mm = BlockReduce(temp_storage).Reduce(val, MinMaxOp{});

    if (threadIdx.x == 0) {
        int block_id = num_proj * node_block_off[node_id] + proj_id * blocks_node + local_bx;
        d_block_min[block_id] = block_mm.minv;
        d_block_max[block_id] = block_mm.maxv;
    }
}

void BlockMinMax_2D_wrap(
    const float* d_col_add_projected,
    const int* d_node_row_off,
    float* d_block_min,
    float* d_block_max,
    const int* d_node_block_off,
    int num_nodes,
    int num_proj,
    int total_blocks)
{
    constexpr int BLOCK = 256;

    dim3 grid(total_blocks, num_proj);
    BlockMinMax_2D<BLOCK><<<grid, BLOCK>>>(d_col_add_projected,
                                           num_proj,
                                           d_node_row_off,
                                           d_node_block_off,
                                           d_block_min,
                                           d_block_max,
                                           num_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK>
__global__ void finalMinMaxReduceAtomic(
    const float* __restrict__ blkMin,
    const float* __restrict__ blkMax,
    const int*   __restrict__ node_block_off,
    float*       __restrict__ outMin,
    float*       __restrict__ outMax,
    int           num_proj)
{
    const int node_id = blockIdx.y;
    const int proj_id = blockIdx.z;
    const int seg = node_id * num_proj + proj_id;

    const int blocks_node = node_block_off[node_id + 1] - node_block_off[node_id];

    int gid_local = blockIdx.x * blockDim.x + threadIdx.x;

    MinMax val;
    val.minv =  INFINITY;
    val.maxv = -INFINITY;

    if (gid_local < blocks_node) {
        int gid = num_proj * node_block_off[node_id] + proj_id * blocks_node + gid_local;
        val.minv = blkMin[gid];
        val.maxv = blkMax[gid];
    }

    using BlockReduce = cub::BlockReduce<MinMax, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    MinMax block_mm =
        BlockReduce(temp).Reduce(val, MinMaxOp{});

    if (threadIdx.x == 0) {
        atomicMinFloat(&outMin[seg], block_mm.minv);
        atomicMaxFloat(&outMax[seg], block_mm.maxv);
    }
}

void finalMinMaxReduceAtomic_wrap(
    float* d_block_min,
    float* d_block_max,
    const int* d_node_block_off,
    float* d_min_vals,
    float* d_max_vals,
    int num_nodes,
    int num_proj,
    int blocks_per_node)
{
    constexpr int BLOCK = 256;
    const int blocks = (blocks_per_node + BLOCK - 1) / BLOCK;
    dim3 grid(blocks, num_nodes, num_proj);
    finalMinMaxReduceAtomic<BLOCK><<<grid, BLOCK>>>(d_block_min,
                                                        d_block_max,
                                                        d_node_block_off,
                                                        d_min_vals,
                                                        d_max_vals,
                                                        num_proj);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK>
__global__ void finalMinMaxReduceAtomic_2D(
    const float* __restrict__ blkMin,
    const float* __restrict__ blkMax,
    const int*   __restrict__ node_block_off,
    const int*   __restrict__ meta_block_off,
    float*       __restrict__ outMin,
    float*       __restrict__ outMax,
    int           num_nodes,
    int           num_proj)
{
    const int proj_id = blockIdx.y;

    const int node_id = cub::UpperBound(meta_block_off, num_nodes + 1, (int)blockIdx.x) - 1;
    const int local_meta_bx = blockIdx.x - meta_block_off[node_id];
    const int blocks_node = node_block_off[node_id + 1] - node_block_off[node_id];

    int gid_local = local_meta_bx * blockDim.x + threadIdx.x;

    MinMax val;
    val.minv =  INFINITY;
    val.maxv = -INFINITY;

    if (gid_local < blocks_node) {
        int gid = num_proj * node_block_off[node_id] + proj_id * blocks_node + gid_local;
        val.minv = blkMin[gid];
        val.maxv = blkMax[gid];
    }

    using BlockReduce = cub::BlockReduce<MinMax, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    MinMax block_mm =
        BlockReduce(temp).Reduce(val, MinMaxOp{});

    if (threadIdx.x == 0) {
        int seg = node_id * num_proj + proj_id;
        atomicMinFloat(&outMin[seg], block_mm.minv);
        atomicMaxFloat(&outMax[seg], block_mm.maxv);
    }
}

void finalMinMaxReduceAtomic_2D_wrap(
    float* d_block_min,
    float* d_block_max,
    const int* d_node_block_off,
    const int* d_meta_block_off,
    float* d_min_vals,
    float* d_max_vals,
    int num_nodes,
    int num_proj,
    int total_meta_blocks)
{
    constexpr int BLOCK = 256;
    dim3 grid(total_meta_blocks, num_proj);
    finalMinMaxReduceAtomic_2D<BLOCK><<<grid, BLOCK>>>(d_block_min,
                                                        d_block_max,
                                                        d_node_block_off,
                                                        d_meta_block_off,
                                                        d_min_vals,
                                                        d_max_vals,
                                                        num_nodes,
                                                        num_proj);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK>
__global__ void RandomBoundaries(
        int                 splits_per_seg,          // NEW  (= num_bins-1)
        int                 rng_seed,                // NEW
        float*              __restrict__ d_min_vals,
        float*              __restrict__ d_max_vals,
        float*              __restrict__ d_candidate_splits
        )
{
    /* 1. identify segment -------------------------------------- */
    const int seg_id  = blockIdx.x;      

    /* 2. process rows ------------------------------------------ */

    if (blockIdx.y == 0) { // only one block per segment should do the following work, so we can use blockIdx.y as a simple filter  
        float mn = d_min_vals[seg_id];
        float mx = d_max_vals[seg_id];

        /* segment could be empty (all rows filtered out) */
        // if (mn > mx) return;               // nothing to generate

        /* one RNG state per thread */
        curandStatePhilox4_32_10_t rng;
        curand_init(rng_seed, seg_id, threadIdx.x, &rng);

        /* Each thread owns one key*/
        float key_local = (threadIdx.x < splits_per_seg)
                        ? mn + curand_uniform(&rng) * (mx - mn)
                        : FLT_MAX;

        /* ---- block-level radix sort ----------------------------------- */
        using BlockSort = cub::BlockRadixSort<float, BLOCK, 1>;
        __shared__ typename BlockSort::TempStorage sort_storage;

        float key_arr[1];
        key_arr[0] = key_local;

        BlockSort(sort_storage).Sort(key_arr);


        /* ---- store the first  splits_per_seg  sorted keys ------------- */
        if (threadIdx.x < splits_per_seg)
            d_candidate_splits[(size_t)seg_id * splits_per_seg + threadIdx.x] =
                key_arr[0];
    }
}

void RandomBoundaries_wrap(const int splits_per_segment,
                            float* d_min_val,
                            float* d_max_val,
                            float* d_sorted_candidate_splits,
                            const int num_segments,
                            const int rng_seed){

    constexpr int BLOCK = 256;
    dim3 grid(num_segments, 1, 1);
    RandomBoundaries<BLOCK><<<grid, BLOCK>>>(
        splits_per_segment,
        rng_seed,
        d_min_val,
        d_max_val,
        d_sorted_candidate_splits
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


template <int BLOCK>
__global__ void BuildHistogramRandomKernel(
        const float*  __restrict__ projected,
        const int*    __restrict__ d_selected_examples,
        const int*    __restrict__ d_labels,
        int*          d_hist_class0,
        int*          d_hist_class1,
        int           num_proj,
        int           num_bins,
        const float*  __restrict__ d_candidate_splits,
        const int*    __restrict__ d_node_row_off,
        int           num_nodes)
{
    
    const int node_id = blockIdx.y;
    const int proj_id = blockIdx.z;
    const int seg_id  = node_id * num_proj + proj_id;

    const int row_start = d_node_row_off[node_id];
    const int rows_node = d_node_row_off[node_id+1] - row_start;
    if (rows_node == 0) return;                       // whole block idle

    extern __shared__ int sh_hist[];
    for (int i = threadIdx.x; i < 2 * num_bins; i += BLOCK){
        sh_hist[i] = 0;}
   
    float* sh_splits = (float*)(sh_hist + 2 * num_bins); // shared memory for candidate splits, placed after the histogram bins

    const float* splits = d_candidate_splits + (size_t)seg_id * (num_bins - 1);

    for (int i = threadIdx.x; i < num_bins - 1; i += BLOCK){
            sh_splits[i] = splits[i];}
    __syncthreads();

  
    /* ----- process one row per thread (if any) --------------- */
    const int gid     = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (gid < rows_node);

    if (active)
    {
        const int r = gid;
        float val = projected[(size_t)row_start * num_proj +
                               (size_t)proj_id * rows_node + r];
        // YDF data load converts labels from {0,1} to {2,1}, this converts back to {0,1} for histogram indexing
        int lbl = 2 - d_labels[d_selected_examples[row_start + r]];
        // int bin = cub::LowerBound(splits, num_bins - 1, val);
        int bin = cub::LowerBound(sh_splits, num_bins - 1, val);
        // int bin = 0;
        // #pragma unroll
        // for (int i = 0; i < num_bins - 1; i++) {
        //     bin += (sh_splits[i] < val);
        // }
        atomicAdd(&sh_hist[lbl * num_bins + bin], 1);
    }
    __syncthreads();            // every thread now reaches the barrier

    /* ----- merge into global histograms ---------------------- */
    // num_bins limited by block size, reconfigure when num_bins > blockDim.x
    if (threadIdx.x < num_bins)
    {
        int off = seg_id * num_bins + threadIdx.x;
        int c0 = sh_hist[threadIdx.x];
        int c1 = sh_hist[num_bins + threadIdx.x];

        if (c0) atomicAdd(&d_hist_class0[off], c0);
        if (c1) atomicAdd(&d_hist_class1[off], c1);
    }
}

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
                                    int blocks_per_node){
    constexpr int BLOCK = 256;
    int sharedMemSize = 2 * num_bins * sizeof(int) +
                        (num_bins - 1) * sizeof(float);
    dim3 grid(blocks_per_node, num_nodes, num_proj);
    BuildHistogramRandomKernel<BLOCK><<<grid, BLOCK, sharedMemSize>>>(
        d_col_add_projected,
        d_selected_examples,
        d_labels,
        d_hist_class0,
        d_hist_class1,
        num_proj,
        num_bins,
        d_sorted_candidate_splits,
        d_node_row_off,
        num_nodes
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK>
__global__ void BuildHistogramRandomKernel_2D(
        const float*  __restrict__ projected,
        const int*    __restrict__ d_selected_examples,
        const int*    __restrict__ d_labels,
        int*          d_hist_class0,
        int*          d_hist_class1,
        int           num_proj,
        int           num_bins,
        const float*  __restrict__ d_candidate_splits,
        const int*    __restrict__ d_node_row_off,
        const int*    __restrict__ d_node_block_off,
        int           num_nodes)
{
    const int proj_id = blockIdx.y;

    const int node_id = cub::UpperBound(d_node_block_off, num_nodes + 1, (int)blockIdx.x) - 1;
    const int local_bx = blockIdx.x - d_node_block_off[node_id];

    const int seg_id = node_id * num_proj + proj_id;

    const int row_start = d_node_row_off[node_id];
    const int rows_node = d_node_row_off[node_id + 1] - row_start;
    if (rows_node == 0) return;

    extern __shared__ int sh_hist[];
    for (int i = threadIdx.x; i < 2 * num_bins; i += BLOCK){
        sh_hist[i] = 0;}

    float* sh_splits = (float*)(sh_hist + 2 * num_bins);

    const float* splits = d_candidate_splits + (size_t)seg_id * (num_bins - 1);

    for (int i = threadIdx.x; i < num_bins - 1; i += BLOCK){
            sh_splits[i] = splits[i];}
    __syncthreads();

    const int gid = local_bx * blockDim.x + threadIdx.x;
    const bool active = (gid < rows_node);

    if (active)
    {
        const int r = gid;
        float val = projected[(size_t)row_start * num_proj +
                               (size_t)proj_id * rows_node + r];
        int lbl = 2 - d_labels[d_selected_examples[row_start + r]];
        int bin = cub::LowerBound(sh_splits, num_bins - 1, val);
        atomicAdd(&sh_hist[lbl * num_bins + bin], 1);
    }
    __syncthreads();

    if (threadIdx.x < num_bins)
    {
        int off = seg_id * num_bins + threadIdx.x;
        int c0 = sh_hist[threadIdx.x];
        int c1 = sh_hist[num_bins + threadIdx.x];

        if (c0) atomicAdd(&d_hist_class0[off], c0);
        if (c1) atomicAdd(&d_hist_class1[off], c1);
    }
}

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
                                    int total_node_blocks){
    constexpr int BLOCK = 256;
    int sharedMemSize = 2 * num_bins * sizeof(int) +
                        (num_bins - 1) * sizeof(float);
    dim3 grid(total_node_blocks, num_proj);
    BuildHistogramRandomKernel_2D<BLOCK><<<grid, BLOCK, sharedMemSize>>>(
        d_col_add_projected,
        d_selected_examples,
        d_labels,
        d_hist_class0,
        d_hist_class1,
        num_proj,
        num_bins,
        d_sorted_candidate_splits,
        d_node_row_off,
        d_node_block_off,
        num_nodes
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}



template<int BLOCK>
__global__ void FindBestEntropySplitKernel(
    const int* hist_class0,
    const int* hist_class1,
    const int num_proj,
    const int num_bins,
    int* hist_class0_prefix,
    int* hist_class1_prefix,
    float* entropy_out_per_bin_per_proj,
    float* best_gain_per_seg,
    int* best_split_per_seg
    ) 
{
    const int node_id = blockIdx.y;
    const int proj_id = blockIdx.z;
    const int seg_id  = node_id * num_proj + proj_id;
    const int bin_id  = threadIdx.x;
    const int base_idx = seg_id * num_bins;
    const bool active = (bin_id < num_bins);

    // ── 1. Prefix sums for histogram bins ───────────────────────────────
    using BlockScan = cub::BlockScan<int, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int prefix0 = active ? hist_class0[base_idx + bin_id] : 0;
    BlockScan(temp_storage).InclusiveSum(prefix0, prefix0);
    __syncthreads();

    int prefix1 = active ? hist_class1[base_idx + bin_id] : 0;
    BlockScan(temp_storage).InclusiveSum(prefix1, prefix1);

    // ── 2. Broadcast totals via shared memory ───────────────────────────
    __shared__ int sh_total_class0;
    __shared__ int sh_total_class1;
    __shared__ float parent_entropy;
    __shared__ float totalf;

    if (bin_id == num_bins - 1) {
        sh_total_class0 = prefix0;
        sh_total_class1 = prefix1;
    }
    __syncthreads();

    const int total_class0 = sh_total_class0;
    const int total_class1 = sh_total_class1;

    if (threadIdx.x == 0) {
        totalf         = float(total_class0) + float(total_class1);
        parent_entropy = entropy(total_class1, total_class0);
    }
    __syncthreads();

    // ── 3. Per-bin entropy gain ─────────────────────────────────────────
    float entropy_gain = -INFINITY;

    if (active) {
        hist_class0_prefix[base_idx + bin_id] = prefix0;
        hist_class1_prefix[base_idx + bin_id] = prefix1;

        const int left_class0  = prefix0;
        const int left_class1  = prefix1;
        const int right_class0 = total_class0 - left_class0;
        const int right_class1 = total_class1 - left_class1;
        const int left_total   = left_class0  + left_class1;
        const int right_total  = right_class0 + right_class1;

        if (left_total != 0 && right_total != 0) {
            float e_left  = entropy(left_class1,  left_class0);
            float e_right = entropy(right_class1, right_class0);

            float weighted =
                ( float(left_total)  * e_left +
                  float(right_total) * e_right ) / fmaxf(totalf, 1.0f);

            entropy_gain = parent_entropy - weighted;
        }

        entropy_out_per_bin_per_proj[base_idx + bin_id] = entropy_gain;
    }

    // ── 4. Block reduce to find best gain/bin ───────────────────────────
    GainBin local_gb;
    local_gb.gain = entropy_gain;
    local_gb.bin  = bin_id;

    using BlockReduceGB = cub::BlockReduce<GainBin, BLOCK>;
    __shared__ typename BlockReduceGB::TempStorage temp_storage_gb;

    GainBin best = BlockReduceGB(temp_storage_gb).Reduce(local_gb, MaxGainBinOp{});

    if (threadIdx.x == 0) {
        best_gain_per_seg[seg_id]  = best.gain;
        best_split_per_seg[seg_id] = best.bin;
    }
}

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
    float* d_out_per_bin_per_proj)
{
    constexpr int BLOCK = 256;
    dim3 grid(1, num_nodes, num_proj);
    FindBestEntropySplitKernel<BLOCK><<<grid, BLOCK>>>(
        d_hist_class0,
        d_hist_class1,
        num_proj,
        num_bins,
        d_prefix_0,
        d_prefix_1,
        d_out_per_bin_per_proj,
        d_best_gain_per_seg,
        d_best_split_per_seg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK>
__global__ void BestProjectionPerNodeKernel(
    const float* __restrict__ best_gain_per_seg,
    const int*   __restrict__ best_split_per_seg,
    int num_proj,
    int num_bins,
    const int*   __restrict__ prefix_0,
    const int*   __restrict__ prefix_1,
    const float* __restrict__ candidate_splits,
    float* __restrict__ best_gain_per_node,
    int*   __restrict__ best_proj_per_node,
    int*   __restrict__ best_split_per_node,
    int*   __restrict__ num_pos_per_node,
    float* __restrict__ best_threshold_per_node
) {
    const int node_id = blockIdx.x;

    GainProjBin local;
    local.gain = -INFINITY;
    local.proj = -1;
    local.bin  = -1;

    for (int proj_id = threadIdx.x; proj_id < num_proj; proj_id += BLOCK) {
        int seg_id = node_id * num_proj + proj_id;
        float g = best_gain_per_seg[seg_id];

        if (g > local.gain || (g == local.gain && proj_id < local.proj)) {
            local.gain = g;
            local.proj = proj_id;
            local.bin  = best_split_per_seg[seg_id];
        }
    }

    using BlockReduce = cub::BlockReduce<GainProjBin, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    GainProjBin best = BlockReduce(temp_storage).Reduce(local, MaxGainProjBinOp{});

    if (threadIdx.x == 0) {
        best_gain_per_node[node_id]  = best.gain;
        best_proj_per_node[node_id]  = best.proj;
        best_split_per_node[node_id] = best.bin;

        if (best.proj < 0) {
            num_pos_per_node[node_id] = 0;
            best_threshold_per_node[node_id] = 0.0f;
            return;
        }

        const int best_seg     = node_id * num_proj + best.proj;
        const int prefix_base  = best_seg * num_bins;
        const int total_0  = prefix_0[prefix_base + num_bins - 1];
        const int total_1  = prefix_1[prefix_base + num_bins - 1];
        const int left_0   = prefix_0[prefix_base + best.bin];
        const int left_1   = prefix_1[prefix_base + best.bin];
        num_pos_per_node[node_id] = total_0 + total_1 - left_0 - left_1;

        const int splits_base = best_seg * (num_bins - 1);
        best_threshold_per_node[node_id] = candidate_splits[splits_base + best.bin];
    }
}

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
                                    const float* d_candidate_splits) {
        dim3 grid(num_nodes, 1, 1);
        int block_size = 32;
        if (num_proj > 32)  block_size = 64;
        if (num_proj > 64)  block_size = 128;
        if (num_proj > 128) block_size = 256;

        #define LAUNCH(B) \
            BestProjectionPerNodeKernel<B><<<grid, B>>>( \
                d_best_gain_per_seg, d_best_split_per_seg, num_proj, num_bins, \
                d_prefix_0, d_prefix_1, d_candidate_splits, \
                d_best_gain_per_node, d_best_proj_per_node, \
                d_best_split_per_node, d_num_pos_per_node, \
                d_best_threshold_per_node)

        switch (block_size) {
            case 32:  LAUNCH(32);  break;
            case 64:  LAUNCH(64);  break;
            case 128: LAUNCH(128); break;
            default:  LAUNCH(256); break;
        }
        #undef LAUNCH
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } 


template<int BLOCK>
__global__ void ColumnAddProjectionKernel_SRDW_1(
        const float*        __restrict__ dataset,
        const int* __restrict__ d_selected_examples,
        float*              __restrict__ projected,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_nodes,
        int                 num_proj,
        int                 num_rows,
        int                 total_rows) // Added num_nodes parameter   
{   
           
    const int node_id = blockIdx.y;
    const int proj_id = blockIdx.z;

    const int global_row = blockIdx.x * blockDim.x + threadIdx.x; // global row index for all nodes
    const int row_stride = gridDim.x * blockDim.x;

    // const int seg_id  = proj_id * num_nodes + node_id;
    const int seg_id  = node_id * num_proj + proj_id; // Changed to node-major order for col_offset

    const int row_start = node_row_off[node_id];
    const int row_end   = node_row_off[node_id + 1];
    const int rows_node = row_end - row_start;
    // Or just pass in the node counts per node? then we read once per thread instead of twice. 

    const int begin = col_offset[seg_id]; // can get rid of this if offset is uniform across all nodes, but for now we keep it general
    const int end   = col_offset[seg_id + 1];

    const int base = row_start * num_proj + proj_id * rows_node;
    for (int r = global_row; r < rows_node; r += row_stride)
    {
        const int ex_idx = d_selected_examples[row_start + r]; //dense read
        if (ex_idx < 0 || ex_idx >= num_rows) continue;
            float sum = 0.0f;

        for (int idx = begin; idx < end; ++idx)
        {
            float w   = flat_weights[idx];
            int   col = flat_col_data[idx];

            // Calculate the index into the dataset array
            const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_rows + ex_idx;
            const float x = dataset[dataset_idx];
            sum += w * x;
            
        }
        // output layout: node-major, then projection, then row
        projected[ base + r ] = sum;
    }
}

template<int BLOCK>
__global__ void ColumnAddProjectionKernel_SRDW_1_2D(
        const float*        __restrict__ dataset,
        const int* __restrict__ d_selected_examples,
        float*              __restrict__ projected,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        const int*          __restrict__ node_block_off,
        int                 num_nodes,
        int                 num_proj,
        int                 num_rows,
        int                 total_rows)
{
    const int proj_id = blockIdx.y;

    const int node_id = cub::UpperBound(node_block_off, num_nodes + 1, (int)blockIdx.x) - 1;
    const int local_bx = blockIdx.x - node_block_off[node_id]; // local block index within the node
    const int blocks_this_node = node_block_off[node_id + 1] - node_block_off[node_id];

    const int local_row = local_bx * blockDim.x + threadIdx.x;
    const int row_stride = blocks_this_node * blockDim.x;

    const int seg_id  = node_id * num_proj + proj_id;

    const int row_start = node_row_off[node_id];
    const int row_end   = node_row_off[node_id + 1];
    const int rows_node = row_end - row_start;

    const int begin = col_offset[seg_id];
    const int end   = col_offset[seg_id + 1];

    const int base = row_start * num_proj + proj_id * rows_node;
    for (int r = local_row; r < rows_node; r += row_stride)
    {
        const int ex_idx = d_selected_examples[row_start + r];
        if (ex_idx < 0 || ex_idx >= num_rows) continue;
            float sum = 0.0f;

        for (int idx = begin; idx < end; ++idx)
        {
            float w   = flat_weights[idx];
            int   col = flat_col_data[idx];

            // Calculate the index into the dataset array
            const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_rows + ex_idx;
            const float x = dataset[dataset_idx];
            sum += w * x;
        }
        projected[ base + r ] = sum;
    }
}

template<int BLOCK>
__global__ void ColumnAddProjectionKernel_SRDW_1_2D_perthread(
        const float*        __restrict__ dataset,
        const int* __restrict__ d_selected_examples,
        float*              __restrict__ projected,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_nodes,
        int                 num_proj,
        int                 num_rows,
        int                 total_rows)
{
    const int proj_id = blockIdx.y;
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= total_rows) return;

    const int node_id = cub::UpperBound(node_row_off, num_nodes + 1, global_tid) - 1;
    const int row_start = node_row_off[node_id];
    const int rows_node = node_row_off[node_id + 1] - row_start;
    const int r = global_tid - node_row_off[node_id];
    if (r >= rows_node) return;

    const int seg_id = node_id * num_proj + proj_id;
    const int begin = col_offset[seg_id];
    const int end   = col_offset[seg_id + 1];

    const int ex_idx = d_selected_examples[row_start + r];
    if (ex_idx < 0 || ex_idx >= num_rows) return;

    float sum = 0.0f;
    for (int idx = begin; idx < end; ++idx)
    {
        float w   = flat_weights[idx];
        int   col = flat_col_data[idx];
        const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_rows + ex_idx;
        sum += w * dataset[dataset_idx];
    }

    const int base = row_start * num_proj + proj_id * rows_node;
    projected[base + r] = sum;
}

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
    int total_rows) {

    constexpr int BLOCK = 256;
    const int blocks = (total_rows + BLOCK - 1) / BLOCK;

    dim3 grid(blocks, num_proj);
    ColumnAddProjectionKernel_SRDW_1_2D_perthread<BLOCK><<<grid, BLOCK>>>(
        d_flat_data, d_selected_examples, d_col_add_projected,
        d_offset, d_flat_projection_col_idx, d_flat_projection_weights,
        d_node_row_off, num_nodes, num_proj, num_rows, total_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

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
    int blocks_per_node) {

    constexpr int BLOCK = 256;

    dim3 grid(blocks_per_node, num_nodes, num_proj);
    ColumnAddProjectionKernel_SRDW_1<BLOCK><<<grid, BLOCK>>>(
        d_flat_data, d_selected_examples, d_col_add_projected,
        d_offset, d_flat_projection_col_idx, d_flat_projection_weights,
        d_node_row_off, num_nodes, num_proj, num_rows, total_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

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
    int total_blocks) {

    constexpr int BLOCK = 256;

    dim3 grid(total_blocks, num_proj);
    ColumnAddProjectionKernel_SRDW_1_2D<BLOCK><<<grid, BLOCK>>>(
        d_flat_data, d_selected_examples, d_col_add_projected,
        d_offset, d_flat_projection_col_idx, d_flat_projection_weights,
        d_node_row_off, d_node_block_off, num_nodes, num_proj, num_rows, total_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK>
__global__ void ColumnAddProjectionKernel_SRDW_2(
        const float*        __restrict__ dataset,
        const int* __restrict__ d_selected_examples,
        float*              __restrict__ projected,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_nodes,
        int                 num_proj,
        int                 num_total_rows) // Added num_nodes parameter   
{   
           
    const int node_id = blockIdx.z;
    const int proj_id = blockIdx.y;

    const int global_row = blockIdx.x * blockDim.x + threadIdx.x; // global row index for all nodes
    const int row_stride = gridDim.x * blockDim.x;

    // const int seg_id  = proj_id * num_nodes + node_id; 
    const int seg_id  = node_id * num_proj + proj_id; // Changed to node-major order for col_offset
 
    const int row_start = node_row_off[node_id];
    const int row_end   = node_row_off[node_id + 1];
    const int rows_node = row_end - row_start;
    // Or just pass in the node counts per node? then we read once per thread instead of twice. 

    const int begin = col_offset[seg_id]; // can get rid of this if offset is uniform across all nodes, but for now we keep it general
    const int end   = col_offset[seg_id + 1]; 

    const int base = row_start * num_proj + proj_id * rows_node;
    for (int r = global_row; r < rows_node; r += row_stride)
    {
        const int ex_idx = d_selected_examples[row_start + r]; //dense read
        if (ex_idx < 0 || ex_idx >= num_total_rows) continue;
        float sum = 0.0f;

        for (int idx = begin; idx < end; ++idx){
            float w   = flat_weights[idx];
            int   col = flat_col_data[idx];
            const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_total_rows + ex_idx; //sparse read
            const float x = dataset[dataset_idx];
            sum += w * x; 
        }
        projected[ base + r ] = sum; //dense write
    }
}


template<int BLOCK>
__global__ void ColumnAddProjectionKernel_DRSW_1(
        const float*        __restrict__ dataset,
        float*              __restrict__ projected,
        const int*          __restrict__ node_ids,
        const int*          __restrict__ node_offsets,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_proj,
        int                 num_total_rows,
        int                 num_nodes,
        int*                node_counts,
        int*                node_row_start_by_row) // Added num_nodes parameter
{
    const int proj_id = blockIdx.y;
    const int row_stride = gridDim.x * blockDim.x;
    const int global_row = blockIdx.x * blockDim.x + threadIdx.x; // global row index for all nodes
    // if (global_row >= num_total_rows) return; // out of bounds check
 
    for (int r = global_row; r < num_total_rows; r += row_stride)
    {
        const int node_id = node_ids[r]; // node_id for this block
        const int row_start = node_row_off[node_id]; // Prefix sum of rows for each node, gives us the starting index in the output for this node
        const int row_end   = node_row_off[node_id + 1];  // node_row_off size is num_nodes + 1, so this is safe
        const int rows_node = row_end - row_start; // number of rows for this node
        // Or just pass in the node counts per node? then we read once per thread instead of twice. 

        const int node_offset = node_offsets[r]; // offset in the output for this thread
        
        // const int seg_id = proj_id * num_nodes + node_id;
        const int seg_id = node_id * num_proj + proj_id; // Changed to node-major order for col_offset

        const int begin = col_offset[seg_id]; // can get rid of this if offset is uniform across all nodes, but for now we keep it general
        const int end   = col_offset[seg_id + 1]; 

        float sum = 0.0f;
        for (int idx = begin; idx < end; ++idx) {
            int   col   = flat_col_data[idx]; // flat_col_data size: num_nodes * num_proj * selected_features_count
            float w     = flat_weights[idx];
            const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_total_rows + r;
            const float x = dataset[dataset_idx];
            sum += w * x;
        }
        // output layout: node-major, then projection, then row
        const int base = row_start * num_proj + proj_id * rows_node;
        projected[base + node_offset] = sum;
    }
}


// void ApplyProjectionBaseline (const float* d_flat_data,
//                               int* d_selected_examples,//selected examples indices
//                               float* d_col_add_projected,
//                               int* d_node_ids,
//                               int* d_node_offsets,
//                               int* d_node_counts,
//                               int* d_node_row_start_by_row,
//                               const int num_proj,
//                               const int num_total_rows,
//                               const int num_segments,
//                               const int num_nodes,
//                               const int num_elems_per_thread,
//                               std::vector<int>& node_start_idx,
//                               std::vector<int>& node_count,
//                               int selected_features_count,
//                               bool alternate,
//                               float* d_min_vals,
//                               float* d_max_vals,
//                               int max_rows_per_node,
//                               int* node_row_off,
//                               int* d_node_row_off,
//                               int* d_offset,
//                               int* d_flat_projection_col_idx,
//                               float* d_flat_projection_weights,
//                               int* d_flat_projection_col_idx_shared,
//                               float* d_flat_projection_weights_shared,
//                               bool verbose)//num_nodes
// {
//     warmup_kernel<<<1,1>>>();
//     cudaDeviceSynchronize();
    
//     constexpr int BLOCK = 256; 
    
//     int rows_per_block = BLOCK * num_elems_per_thread;
//     int blocks_per_all_nodes_per_proj = (num_total_rows + rows_per_block - 1) / rows_per_block;
//     int blocks_per_node = (max_rows_per_node + rows_per_block - 1) / rows_per_block;

//     if (alternate) {
//         // Launch the alternative kernel that uses node_ids and node_offsets
//         dim3 gridAP_DRSW_1(blocks_per_all_nodes_per_proj, num_proj);
//         auto start_kernel_DRSW_1 = std::chrono::steady_clock::now();
//         ColumnAddProjectionKernel_DRSW_1<BLOCK><<<gridAP_DRSW_1, BLOCK>>>(
//             d_flat_data,                     // your original device dataset
//             d_col_add_projected,             // output buffer (device)
//             d_node_ids,
//             d_node_offsets,
//             d_offset,                        // column offsets
//             d_flat_projection_col_idx,
//             d_flat_projection_weights,
//             d_node_row_off,                  // per-node row prefix
//             num_proj,
//             num_total_rows,
//             num_nodes,
//             d_node_counts,
//             d_node_row_start_by_row); // Pass num_nodes to the kernel
//         cudaDeviceSynchronize();
//         auto end_kernel_DRSW_1 = std::chrono::steady_clock::now();
//         std::chrono::duration<double, std::milli> kernel_DRSW_1_duration = end_kernel_DRSW_1 - start_kernel_DRSW_1;
//         std::cout << "Time taken for ColumnAddProjectionKernel_DRSW_1: " << kernel_DRSW_1_duration.count() << " ms" << std::endl;

//     } 

//     else { 

//         dim3 gridAP_SRDW_1(blocks_per_node, num_nodes, num_proj); // 3D grid with node_id as the z-dimension
//         auto start_kernel_SRDW_1 = std::chrono::steady_clock::now();
//         ColumnAddProjectionKernel_SRDW_1<BLOCK><<<gridAP_SRDW_1, BLOCK>>>(
//             d_flat_data,                     // your original device dataset
//             d_selected_examples,             // flat rows (all nodes)
//             d_col_add_projected,             // output buffer (device)
//             d_offset,                        // column offsets
//             d_flat_projection_col_idx,
//             d_flat_projection_weights,
//             d_node_row_off,                  // per-node row prefix
//             num_nodes,
//             num_proj,
//             num_total_rows); 
//         cudaDeviceSynchronize();
//         auto end_kernel_SRDW_1 = std::chrono::steady_clock::now();
//         std::chrono::duration<double, std::milli> kernel_SRDW_1_duration = end_kernel_SRDW_1 - start_kernel_SRDW_1;
//         std::cout << "Time taken for ColumnAddProjectionKernel_SRDW_1: " << kernel_SRDW_1_duration.count() << " ms" << std::endl;


//         dim3 gridAP_SRDW_2(blocks_per_node, num_proj, num_nodes); // 3D grid with node_id as the z-dimension
//         auto start_kernel_SRDW_2 = std::chrono::steady_clock::now();
//         ColumnAddProjectionKernel_SRDW_2<BLOCK><<<gridAP_SRDW_2, BLOCK>>>(
//             d_flat_data,                     // your original device dataset
//             d_selected_examples,             // flat rows (all nodes)
//             d_col_add_projected,             // output buffer (device)
//             d_offset,                        // column offsets
//             d_flat_projection_col_idx,
//             d_flat_projection_weights,
//             d_node_row_off,                  // per-node row prefix
//             num_nodes,
//             num_proj,
//             num_total_rows); 
//         cudaDeviceSynchronize();
//         auto end_kernel_SRDW_2 = std::chrono::steady_clock::now();
//         std::chrono::duration<double, std::milli> kernel_SRDW_2_duration = end_kernel_SRDW_2 - start_kernel_SRDW_2;
//         std::cout << "Time taken for ColumnAddProjectionKernel_SRDW_2: " << kernel_SRDW_2_duration.count() << " ms" << std::endl;

//     }
// }

