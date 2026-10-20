#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/thread/thread_search.cuh>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>  // DEDUP_BOOTSTRAP: for thrust::unique
#include <cfloat>
#include <cstring>
#include <iostream>
#include <vector>

#include "yggdrasil_decision_forests/learner/decision_tree/gpu_timer.h"

static double s_sad_malloc = 0, s_sad_flags = 0, s_sad_prefix = 0,
              s_sad_scatter = 0, s_sad_d2h_scatter = 0, s_sad_d2h_counts = 0,
              s_sad_d2d_copy = 0, s_sad_free = 0;

static double s_tree_sad_malloc = 0, s_tree_sad_flags = 0, s_tree_sad_prefix = 0,
              s_tree_sad_scatter = 0, s_tree_sad_d2h_scatter = 0, s_tree_sad_d2h_counts = 0,
              s_tree_sad_d2d_copy = 0, s_tree_sad_free = 0;

static double s_drsw_compute = 0, s_drsw_proj = 0;
static double s_tree_drsw_compute = 0, s_tree_drsw_proj = 0;

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

// ── Sampling WITHOUT replacement ───────────────────────────────────────────
// Identity selection: d_selected_examples[i] = i (the whole dataset, once).
// Used for bootstrap_training_dataset=false, and for sampling_with_replacement
// =false with bootstrap_size_ratio>=1 (all examples per YDF semantics).
__global__ void IotaSelectedExamplesKernel(
    int* __restrict__ d_selected_examples,
    const int num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    d_selected_examples[tid] = tid;
}

// Per-row random key + its index, for a without-replacement SUBSET (ratio<1):
// sort indices by key and take the first k — a uniform sample without repeats.
__global__ void FillRandomKeysAndIotaKernel(
    unsigned int* __restrict__ d_keys,
    int*          __restrict__ d_idx,
    const int num_rows,
    const unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    curandState state;
    curand_init(seed, tid, 0, &state);
    d_keys[tid] = curand(&state);
    d_idx[tid]  = tid;
}

// How many examples a tree draws, honoring the RF config:
//   bootstrap_training_dataset=false      -> num_rows (all examples, once)
//   sampling_with_replacement=true        -> round(ratio*N)   (may exceed N)
//   sampling_with_replacement=false       -> min(N, round(ratio*N))  (<= N)
static int SelectedExamplesCount(
    int num_rows,
    const yggdrasil_decision_forests::model::random_forest::proto::RandomForestTrainingConfig& rf_config)
{
    const int64_t raw = std::max<int64_t>(
        1, static_cast<int64_t>(static_cast<double>(num_rows) * rf_config.bootstrap_size_ratio()));
    if (!rf_config.bootstrap_training_dataset()) return num_rows;
    if (rf_config.sampling_with_replacement()) return static_cast<int>(raw);
    return static_cast<int>(std::min<int64_t>(num_rows, raw));
}

// Fill d_selected_examples[0..num_samples) and leave it sorted ascending
// (DeduplicateBootstrap and the SRDW/node-id consumers expect sorted indices).
static absl::Status FillSelectedExamples(
    int* d_selected_examples,
    int num_samples,
    int num_rows,
    bool with_replacement,
    unsigned long long seed)
{
    constexpr int BLOCK = 256;
    bool need_sort = true;

    if (with_replacement) {
        const int blocks = (num_samples + BLOCK - 1) / BLOCK;
        SampleTrainingExamplesKernel<<<blocks, BLOCK>>>(
            d_selected_examples, num_samples, num_rows, seed);
    } else if (num_samples >= num_rows) {
        // Without replacement, full dataset -> identity (already sorted).
        const int blocks = (num_rows + BLOCK - 1) / BLOCK;
        IotaSelectedExamplesKernel<<<blocks, BLOCK>>>(d_selected_examples, num_rows);
        need_sort = false;
    } else {
        // Without replacement, strict subset -> random-key sort, take first k.
        unsigned int* d_keys = nullptr;
        int* d_idx = nullptr;
        CUDA_CHECK(cudaMalloc(&d_keys, static_cast<size_t>(num_rows) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_idx,  static_cast<size_t>(num_rows) * sizeof(int)));
        const int blocks = (num_rows + BLOCK - 1) / BLOCK;
        FillRandomKeysAndIotaKernel<<<blocks, BLOCK>>>(d_keys, d_idx, num_rows, seed);
        CUDA_CHECK(cudaGetLastError());
        thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_keys),
                            thrust::device_ptr<unsigned int>(d_keys) + num_rows,
                            thrust::device_ptr<int>(d_idx));
        CUDA_CHECK(cudaMemcpy(d_selected_examples, d_idx,
                              static_cast<size_t>(num_samples) * sizeof(int),
                              cudaMemcpyDeviceToDevice));
        cudaFree(d_keys);
        cudaFree(d_idx);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (need_sort) {
        thrust::device_ptr<int> d_ptr(d_selected_examples);
        thrust::sort(d_ptr, d_ptr + num_samples);
    }
    return absl::OkStatus();
}

absl::Status SampleTrainingExamplesKernel_wrap(
    const int num_rows,
    const yggdrasil_decision_forests::model::random_forest::proto::RandomForestTrainingConfig& rf_config,
    std::optional<double> bootstrap_size_ratio_factor,
    int* d_selected_examples,
    std::vector<yggdrasil_decision_forests::dataset::UnsignedExampleIdx>* selected_examples,
    unsigned long long seed)
{
    const int num_samples = SelectedExamplesCount(num_rows, rf_config);
    const bool with_replacement =
        rf_config.bootstrap_training_dataset() && rf_config.sampling_with_replacement();

    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_samples * sizeof(int)));

    auto st = FillSelectedExamples(
        d_selected_examples, num_samples, num_rows, with_replacement, seed);
    if (!st.ok()) return st;

    selected_examples->resize(num_samples);
    CUDA_CHECK(cudaMemcpy(selected_examples->data(), d_selected_examples,
        num_samples * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_selected_examples));
    return absl::OkStatus();
}

absl::Status SampleTrainingExamplesKernel_No_D2H_wrap(
    const int num_rows,
    const yggdrasil_decision_forests::model::random_forest::proto::RandomForestTrainingConfig& rf_config,
    std::optional<double> bootstrap_size_ratio_factor,
    int*& d_selected_examples,
    int& num_samples_out,
    unsigned long long seed)
{
    const int num_samples = SelectedExamplesCount(num_rows, rf_config);
    const bool with_replacement =
        rf_config.bootstrap_training_dataset() && rf_config.sampling_with_replacement();

    num_samples_out = num_samples;

    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_samples * sizeof(int)));

    auto st = FillSelectedExamples(
        d_selected_examples, num_samples, num_rows, with_replacement, seed);
    if (!st.ok()) return st;

    return absl::OkStatus();
}

// --- DEDUP_BOOTSTRAP: deduplicate sorted bootstrap indices and build
//     per-row duplicate counts (d_ex_dup_counts[row] = how many times
//     that row was sampled).  Allows toggling back to duplicated mode
//     by not defining DEDUP_BOOTSTRAP. ---

__global__ void ComputeBootstrapDupCountsKernel(
    const int* __restrict__ d_bootstrap, //d_selected_examples
    int* __restrict__ d_ex_dup_counts,
    int num_samples)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;
    atomicAdd(&d_ex_dup_counts[d_bootstrap[tid]], 1);
}

absl::Status DeduplicateBootstrap_wrap(
    int*& d_selected_examples,
    int*& d_ex_dup_counts,
    int& num_unique_out,
    int num_samples,
    int num_rows)
{
    // 1. Allocate and zero the per-row dup counts array (size = num_rows)
    CUDA_CHECK(cudaMalloc(&d_ex_dup_counts, num_rows * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ex_dup_counts, 0, num_rows * sizeof(int)));

    // 2. Count how many times each row appears in the bootstrap
    constexpr int BLOCK = 256;
    const int blocks = (num_samples + BLOCK - 1) / BLOCK;
    ComputeBootstrapDupCountsKernel<<<blocks, BLOCK>>>(
        d_selected_examples, d_ex_dup_counts, num_samples);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // 3. Deduplicate the sorted selected_examples in-place
    thrust::device_ptr<int> d_ptr(d_selected_examples);
    auto new_end = thrust::unique(d_ptr, d_ptr + num_samples);
    num_unique_out = static_cast<int>(new_end - d_ptr);

    return absl::OkStatus();
}

// Direct scatter: reads from d_read_buf, writes to d_selected_examples at
// compacted child_offsets. Leaf nodes (d_child_offsets < 0) are skipped entirely.
// Neg fills from front, pos fills from back within each node's allocation.
__global__ void DirectScatterFusedAllDepthKernel_1D(
    const float* __restrict__ d_col_add_projected,
    const int* __restrict__ d_read_buf,
    int* __restrict__ d_selected_examples,
    const int* __restrict__ d_best_proj_per_node,
    const float* __restrict__ d_best_threshold_per_node,
    const int* __restrict__ d_node_row_off,
    const int* __restrict__ d_node_block_off,
    const int* __restrict__ d_child_offsets,
    const int* __restrict__ d_labels,
#ifdef DEDUP_BOOTSTRAP
    const int* __restrict__ d_ex_dup_counts,  // ADDED: per-row bootstrap multiplicity
#endif
    int* __restrict__ d_class_counts,
    const int num_classes,
    int* __restrict__ d_atomic_counters,
    int num_proj,
    int num_nodes)
{
    const int node_id = cub::UpperBound(d_node_block_off, num_nodes + 1, (int)blockIdx.x) - 1;
    const int child_off = d_child_offsets[node_id];
    if (child_off < 0) return;

    const int local_bx = blockIdx.x - d_node_block_off[node_id];
    const int rows_start = d_node_row_off[node_id];
    const int rows_node = d_node_row_off[node_id + 1] - rows_start;
    const int r = local_bx * blockDim.x + threadIdx.x;
    if (r >= rows_node) return;

    int best_projection_index = d_best_proj_per_node[node_id];
    int side = 0;
    if (best_projection_index >= 0) {
        const long long base = (long long)rows_start * num_proj + (long long)best_projection_index * rows_node;
        float val = d_col_add_projected[base + r];
        side = val >= d_best_threshold_per_node[node_id] ? 1 : 0;
    }

    int example_idx = d_read_buf[rows_start + r];
    if (side) {
        int pos_idx = atomicAdd(&d_atomic_counters[node_id], 1);
        d_selected_examples[child_off + rows_node - 1 - pos_idx] = example_idx;
    } else {
        int neg_idx = atomicAdd(&d_atomic_counters[num_nodes + node_id], 1);
        d_selected_examples[child_off + neg_idx] = example_idx;
    }

    int cls = d_labels[example_idx];
#ifdef DEDUP_BOOTSTRAP //only class counts matter with duplicates
    atomicAdd(&d_class_counts[node_id * 2 * num_classes + side * num_classes + cls],
              d_ex_dup_counts[example_idx]);
#else
    atomicAdd(&d_class_counts[node_id * 2 * num_classes + side * num_classes + cls], 1);
#endif
}
template <int BLOCK_SIZE>
__global__ void StablePartitionCountKernel(
    const float* __restrict__ d_col_add_projected,
    const int* __restrict__ d_best_proj_per_node,
    const float* __restrict__ d_best_threshold_per_node,
    const int* __restrict__ d_node_row_off,
    const int* __restrict__ d_node_block_off,
    const int* __restrict__ d_child_offsets,
    int* __restrict__ d_block_right_counts,
    int num_proj,
    int num_nodes)
{
    using BlockReduce = cub::BlockReduce<int, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage reduce_temp;

    const int node_id = cub::UpperBound(d_node_block_off, num_nodes + 1, (int)blockIdx.x) - 1;
    if (d_child_offsets[node_id] < 0) {
        if (threadIdx.x == 0) d_block_right_counts[blockIdx.x] = 0;
        return;
    }

    const int local_bx = blockIdx.x - d_node_block_off[node_id];
    const int rows_start = d_node_row_off[node_id];
    const int rows_node  = d_node_row_off[node_id + 1] - rows_start;
    const int r = local_bx * BLOCK_SIZE + threadIdx.x;

    int side = 0;
    if (r < rows_node) {
        int best_proj = d_best_proj_per_node[node_id];
        if (best_proj >= 0) {
            const long long base = (long long)rows_start * num_proj
                                 + (long long)best_proj * rows_node;
            side = d_col_add_projected[base + r]
                   >= d_best_threshold_per_node[node_id] ? 1 : 0;
        }
    }

    int total_right = BlockReduce(reduce_temp).Sum(side);
    if (threadIdx.x == 0)
        d_block_right_counts[blockIdx.x] = total_right;
}

template <int BLOCK_SIZE>
__global__ void StablePartitionScatterKernel(
    const float* __restrict__ d_col_add_projected,
    const int* __restrict__ d_read_buf,
    int* __restrict__ d_selected_examples,
    const int* __restrict__ d_best_proj_per_node,
    const float* __restrict__ d_best_threshold_per_node,
    const int* __restrict__ d_node_row_off,
    const int* __restrict__ d_node_block_off,
    const int* __restrict__ d_child_offsets,
    const int* __restrict__ d_labels,
#ifdef DEDUP_BOOTSTRAP
    const int* __restrict__ d_ex_dup_counts,
#endif
    int* __restrict__ d_class_counts,
    const int num_classes,
    const int* __restrict__ d_block_right_prefix,
    int num_proj,
    int num_nodes)
{
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    __shared__ int s_node_prefix_right;

    const int node_id = cub::UpperBound(d_node_block_off, num_nodes + 1, (int)blockIdx.x) - 1;
    const int child_off = d_child_offsets[node_id];
    if (child_off < 0) return;

    const int node_block_start = d_node_block_off[node_id];
    const int local_bx = blockIdx.x - node_block_start;
    const int rows_start = d_node_row_off[node_id];
    const int rows_node  = d_node_row_off[node_id + 1] - rows_start;
    const int r = local_bx * BLOCK_SIZE + threadIdx.x;
    const bool valid = (r < rows_node);

    if (threadIdx.x == 0)
        s_node_prefix_right = d_block_right_prefix[node_block_start];
    __syncthreads();

    int side = 0;
    if (valid) {
        int best_proj = d_best_proj_per_node[node_id];
        if (best_proj >= 0) {
            const long long base = (long long)rows_start * num_proj
                                 + (long long)best_proj * rows_node;
            side = d_col_add_projected[base + r]
                   >= d_best_threshold_per_node[node_id] ? 1 : 0;
        }
    }

    int right_flag = valid ? side : 0;
    int right_prefix_local;
    BlockScan(scan_temp).ExclusiveSum(right_flag, right_prefix_local);
    __syncthreads();

    if (!valid) return;

    // Per-node offsets: subtract this node's starting prefix so counts
    // are relative to the node, not the global block array.
    int blocks_before_right = d_block_right_prefix[blockIdx.x] - s_node_prefix_right;
    int valid_before = local_bx * BLOCK_SIZE;
    int blocks_before_left  = valid_before - blocks_before_right;

    int example_idx = d_read_buf[rows_start + r];
    if (side) {
        int pos = blocks_before_right + right_prefix_local;
        d_selected_examples[child_off + rows_node - 1 - pos] = example_idx;
    } else {
        int left_prefix_local = threadIdx.x - right_prefix_local;
        int pos = blocks_before_left + left_prefix_local;
        d_selected_examples[child_off + pos] = example_idx;
    }

    int cls = d_labels[example_idx];
#ifdef DEDUP_BOOTSTRAP
    atomicAdd(&d_class_counts[node_id * 2 * num_classes + side * num_classes + cls],
              d_ex_dup_counts[example_idx]);
#else
    atomicAdd(&d_class_counts[node_id * 2 * num_classes + side * num_classes + cls], 1);
#endif
}
// ============================================================================
// GPU Projection Sampling (guarded by GPU_SAMPLE_PROJ)
// Two-pass: count features per projection, prefix sum → CSR offsets, fill data.
// ============================================================================
#ifdef GPU_SAMPLE_PROJ

__device__ int sample_binomial(curandStatePhilox4_32_10_t* state, int n, float p) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (curand_uniform(state) < p) count++;
    }
    return count;
}

template <int BLOCK>
__global__ void SampleProjectionCountKernel(
    int* __restrict__ d_proj_counts,
    int num_features,
    int num_segments,
    int num_proj,
    float projection_density,
    int max_num_features,
    unsigned long long seed)
{
    int seg_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg_id >= num_segments) return;

    // Segments are laid out as seg_id = node_id * num_proj + proj_id. Under
    // SYMMETRIC_PROJ every node shares the same set of projections, so the
    // curand stream keys on proj_id alone; otherwise each (node, proj) segment
    // gets its own stream.
#ifdef SYMMETRIC_PROJ
    const unsigned long long subseq = (unsigned long long)(seg_id % num_proj);
#else
    const unsigned long long subseq = (unsigned long long)seg_id;
#endif
    curandStatePhilox4_32_10_t state;
    curand_init(seed, subseq, 0ULL, &state);

    int count = sample_binomial(&state, num_features, projection_density);
    if (count <= 0) count = 1;
    if (max_num_features > 0 && count > max_num_features) count = max_num_features;

    d_proj_counts[seg_id] = count;
}

template <int BLOCK, int MAX_K = 64>
__global__ void SampleProjectionFillKernel(
    int* __restrict__ d_flat_projection_col_idx,
    int* __restrict__ d_flat_projection_attr_idx,
    float* __restrict__ d_flat_projection_weights,
    const int* __restrict__ d_offset,
    const int* __restrict__ d_numerical_features,
    const int* __restrict__ d_attr_to_flat_col,
    const float* __restrict__ d_norm_factors,
    int num_features,
    int num_segments,
    int num_proj,
    float projection_density,
    int max_num_features,
    unsigned long long seed,
    int weight_type)
{
    int seg_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg_id >= num_segments) return;

    // Match SampleProjectionCountKernel's stream keying: under SYMMETRIC_PROJ
    // all nodes sharing a proj_id draw the identical feature set + weights.
#ifdef SYMMETRIC_PROJ
    const unsigned long long subseq = (unsigned long long)(seg_id % num_proj);
#else
    const unsigned long long subseq = (unsigned long long)seg_id;
#endif
    curandStatePhilox4_32_10_t state;
    curand_init(seed, subseq, 0ULL, &state);

    int raw_count = sample_binomial(&state, num_features, projection_density);
    bool forced_single = (raw_count <= 0);
    if (forced_single) raw_count = 1;
    int count = raw_count;
    if (max_num_features > 0 && count > max_num_features) count = max_num_features;

    // Floyd's algorithm: select raw_count features from [0, num_features)
    int picked[MAX_K];
    int k = min(raw_count, (int)MAX_K);
    int picked_count = 0;
    for (int j = num_features - k; j < num_features; j++) {
        float u = curand_uniform(&state);
        int t = (int)(u * (j + 1));
        if (t > j) t = j;
        bool found = false;
        for (int i = 0; i < picked_count; i++) {
            if (picked[i] == t) { found = true; break; }
        }
        if (found)
            picked[picked_count] = j;
        else
            picked[picked_count] = t;
        picked_count++;
    }

    // If max_num_features requires sub-sampling from the picked set
    int final_indices[MAX_K];
    int final_count = count;
    if (max_num_features > 0 && raw_count > max_num_features) {
        int sub_count = 0;
        for (int j = raw_count - max_num_features; j < raw_count; j++) {
            float u = curand_uniform(&state);
            int t = (int)(u * (j + 1));
            if (t > j) t = j;
            bool found = false;
            for (int i = 0; i < sub_count; i++) {
                if (final_indices[i] == t) { found = true; break; }
            }
            if (found)
                final_indices[sub_count] = j;
            else
                final_indices[sub_count] = t;
            sub_count++;
        }
        for (int i = 0; i < final_count; i++)
            final_indices[i] = picked[final_indices[i]];
    } else {
        for (int i = 0; i < final_count; i++)
            final_indices[i] = picked[i];
    }

    int write_base = d_offset[seg_id];

    for (int i = 0; i < final_count; i++) {
        int feat_pos = final_indices[i];
        int ydf_attr = d_numerical_features[feat_pos];
        int flat_col = d_attr_to_flat_col[ydf_attr];

        float weight;
        if (forced_single || final_count == 1) {
            weight = 1.0f;
        } else {
            float u = curand_uniform(&state) * 2.0f - 1.0f;
            weight = (weight_type == 1) ? ((u >= 0.0f) ? 1.0f : -1.0f) : u;
        }

        weight *= d_norm_factors[feat_pos];

        d_flat_projection_col_idx[write_base + i] = flat_col;
        d_flat_projection_attr_idx[write_base + i] = ydf_attr;
        d_flat_projection_weights[write_base + i] = weight;
    }
}

size_t SampleProjectionsCubTempBytes(int num_segments)
{
    // ExclusiveSum's temp requirement depends only on the item count and is
    // monotonic in it, so sizing at the max num_segments covers every depth.
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    int* d_dummy = nullptr;  // null is fine for the sizing query
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes,
        d_dummy, d_dummy, num_segments + 1);
    return temp_bytes;
}

void SampleProjectionsOnGPU_wrap(
    int* d_offset,
    int* d_flat_projection_col_idx,
    int* d_flat_projection_attr_idx,
    float* d_flat_projection_weights,
    const int* d_numerical_features,
    const int* d_attr_to_flat_col,
    const float* d_norm_factors,
    int* d_proj_counts,
    void* d_cub_temp,
    size_t cub_temp_bytes,
    int num_features,
    int num_nodes,
    int num_proj,
    float projection_density,
    int max_num_features,
    unsigned long long seed,
    int weight_type)
{
    const int num_segments = num_nodes * num_proj;
    constexpr int BLOCK = 256;
    const int grid = (num_segments + BLOCK - 1) / BLOCK;

    // d_proj_counts and d_cub_temp are persistent buffers owned by the caller
    // (allocated once before the depth loop). The count kernel writes seg_id in
    // [0, num_segments); zero the trailing element the prefix sum reads.
    CUDA_CHECK(cudaMemset(d_proj_counts + num_segments, 0, sizeof(int)));

    SampleProjectionCountKernel<BLOCK><<<grid, BLOCK>>>(
        d_proj_counts, num_features, num_segments, num_proj,
        projection_density, max_num_features, seed);

    size_t temp_bytes = cub_temp_bytes;
    cub::DeviceScan::ExclusiveSum(d_cub_temp, temp_bytes,
        d_proj_counts, d_offset, num_segments + 1);

    SampleProjectionFillKernel<BLOCK><<<grid, BLOCK>>>(
        d_flat_projection_col_idx, d_flat_projection_attr_idx,
        d_flat_projection_weights, d_offset,
        d_numerical_features, d_attr_to_flat_col, d_norm_factors,
        num_features, num_segments, num_proj, projection_density,
        max_num_features, seed, weight_type);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

#endif  // GPU_SAMPLE_PROJ
void PrintAndResetSplitAllDepthTimers(int depth) {
    double total = s_sad_malloc + s_sad_flags + s_sad_prefix + s_sad_scatter
                 + s_sad_d2h_scatter + s_sad_d2h_counts + s_sad_d2d_copy + s_sad_free;
#ifdef VERBOSE_DEPTH
    std::cerr << "  SplitAllDepth D" << depth << ":";
    if (s_sad_malloc > 0)      std::cerr << " malloc=" << s_sad_malloc;
    if (s_sad_flags > 0)       std::cerr << " flags_kernel=" << s_sad_flags;
    if (s_sad_prefix > 0)      std::cerr << " prefix_sum=" << s_sad_prefix;
    if (s_sad_scatter > 0)     std::cerr << " scatter_kernel=" << s_sad_scatter;
    if (s_sad_d2h_scatter > 0) std::cerr << " d2h_scatter=" << s_sad_d2h_scatter;
    if (s_sad_d2h_counts > 0)  std::cerr << " d2h_counts=" << s_sad_d2h_counts;
    if (s_sad_d2d_copy > 0)    std::cerr << " d2d_copy=" << s_sad_d2d_copy;
    if (s_sad_free > 0)        std::cerr << " free=" << s_sad_free;
    std::cerr << " SUM=" << total << "ms" << std::endl;
#endif
    s_tree_sad_malloc += s_sad_malloc;
    s_tree_sad_flags += s_sad_flags;
    s_tree_sad_prefix += s_sad_prefix;
    s_tree_sad_scatter += s_sad_scatter;
    s_tree_sad_d2h_scatter += s_sad_d2h_scatter;
    s_tree_sad_d2h_counts += s_sad_d2h_counts;
    s_tree_sad_d2d_copy += s_sad_d2d_copy;
    s_tree_sad_free += s_sad_free;
    s_sad_malloc = s_sad_flags = s_sad_prefix = s_sad_scatter = 0;
    s_sad_d2h_scatter = s_sad_d2h_counts = s_sad_d2d_copy = s_sad_free = 0;
}

double GetSplitAllDepthTreeFreeMs() { return s_tree_sad_free; }

void PrintAndResetSplitAllDepthTreeTimers() {
    double total = s_tree_sad_malloc + s_tree_sad_flags + s_tree_sad_prefix + s_tree_sad_scatter
                 + s_tree_sad_d2h_scatter + s_tree_sad_d2h_counts + s_tree_sad_d2d_copy + s_tree_sad_free;
    std::cerr << "  SplitAllDepth TREE TOTAL:";
    if (s_tree_sad_malloc > 0)      std::cerr << " malloc=" << s_tree_sad_malloc;
    if (s_tree_sad_flags > 0)       std::cerr << " flags_kernel=" << s_tree_sad_flags;
    if (s_tree_sad_prefix > 0)      std::cerr << " prefix_sum=" << s_tree_sad_prefix;
    if (s_tree_sad_scatter > 0)     std::cerr << " scatter_kernel=" << s_tree_sad_scatter;
    if (s_tree_sad_d2h_scatter > 0) std::cerr << " d2h_scatter=" << s_tree_sad_d2h_scatter;
    if (s_tree_sad_d2h_counts > 0)  std::cerr << " d2h_counts=" << s_tree_sad_d2h_counts;
    if (s_tree_sad_d2d_copy > 0)    std::cerr << " d2d_copy=" << s_tree_sad_d2d_copy;
    if (s_tree_sad_free > 0)        std::cerr << " free=" << s_tree_sad_free;
    std::cerr << " SUM=" << total << "ms" << std::endl;
    s_tree_sad_malloc = s_tree_sad_flags = s_tree_sad_prefix = s_tree_sad_scatter = 0;
    s_tree_sad_d2h_scatter = s_tree_sad_d2h_counts = s_tree_sad_d2d_copy = s_tree_sad_free = 0;
}

void PrintAndResetDRSWTimers(int depth) {
    double total = s_drsw_compute + s_drsw_proj;
#ifdef VERBOSE_DEPTH
    std::cerr << "  DRSW D" << depth << ":"
              << " setup_arrays=" << s_drsw_compute
              << " projection_kernel=" << s_drsw_proj
              << " SUM=" << total << "ms" << std::endl;
#endif
    s_tree_drsw_compute += s_drsw_compute;
    s_tree_drsw_proj    += s_drsw_proj;
    s_drsw_compute = s_drsw_proj = 0;
}

double GetDRSWTreeSetupMs() { return s_tree_drsw_compute; }
double GetDRSWTreeProjMs() { return s_tree_drsw_proj; }

void PrintAndResetDRSWTreeTimers() {
    double total = s_tree_drsw_compute + s_tree_drsw_proj;
    std::cerr << "  DRSW TREE TOTAL:"
              << " setup_arrays=" << s_tree_drsw_compute
              << " projection_kernel=" << s_tree_drsw_proj
              << " SUM=" << total << "ms" << std::endl;
    s_tree_drsw_compute = s_tree_drsw_proj = 0;
}

// --- Fused No D2H version: single kernel, no prefix sum ---
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
#ifdef DEDUP_BOOTSTRAP
                const int* d_ex_dup_counts,  // ADDED: per-row bootstrap multiplicity
#endif
                const int num_classes,
                std::vector<std::vector<int>>& pos_class_counts,
                std::vector<std::vector<int>>& neg_class_counts,
                std::vector<int>& pos_counts,
                std::vector<int>& neg_counts,
                const std::vector<bool>& node_has_split,
                std::vector<int>& child_offsets)
{
    constexpr int BLOCK = 256;

#if defined(D2D_DIRECT)
    // --- Variant D: direct scatter — no temp buffer, no D2D compaction ---
    auto t0 = now();

    // Temp read buffer (single contiguous D2D instead of 2*total_rows temp)
    int* d_read_buf;
    CUDA_CHECK(cudaMalloc(&d_read_buf, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_read_buf, d_selected_examples,
                          total_rows * sizeof(int), cudaMemcpyDeviceToDevice));

    const int counts_size = num_nodes * 2 * num_classes;
    int* d_class_counts;
    CUDA_CHECK(cudaMalloc(&d_class_counts, counts_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_class_counts, 0, counts_size * sizeof(int)));

    int* d_atomic_counters;
    CUDA_CHECK(cudaMalloc(&d_atomic_counters, 2 * num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_atomic_counters, 0, 2 * num_nodes * sizeof(int)));

    // Precompute child_offsets (uses node sizes, not kernel output)
    child_offsets.assign(num_nodes, -1);
    int write_offset = 0;
    for (int n = 0; n < num_nodes; ++n) {
      if (!node_has_split[n]) continue;
      child_offsets[n] = write_offset;
      write_offset += node_row_off[n + 1] - node_row_off[n];
    }

    int* d_child_offsets;
    CUDA_CHECK(cudaMalloc(&d_child_offsets, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_child_offsets, child_offsets.data(),
                          num_nodes * sizeof(int), cudaMemcpyHostToDevice));

    auto t1 = now();

    DirectScatterFusedAllDepthKernel_1D<<<total_node_blocks, BLOCK>>>(
        d_col_add_projected,
        d_read_buf,
        d_selected_examples,
        d_best_proj_per_node,
        d_best_threshold_per_node,
        d_node_row_off,
        d_node_block_off,
        d_child_offsets,
        d_labels,
#ifdef DEDUP_BOOTSTRAP
        d_ex_dup_counts,
#endif
        d_class_counts,
        num_classes,
        d_atomic_counters,
        num_proj,
        num_nodes);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    auto t2 = now();

    // D2H: class counts + atomic counters
    std::vector<int> h_class_counts(counts_size);
    CUDA_CHECK(cudaMemcpy(h_class_counts.data(), d_class_counts,
                          counts_size * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_counters(2 * num_nodes);
    CUDA_CHECK(cudaMemcpy(h_counters.data(), d_atomic_counters,
                          2 * num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

    pos_class_counts.resize(num_nodes);
    neg_class_counts.resize(num_nodes);
    pos_counts.resize(num_nodes);
    neg_counts.resize(num_nodes);
    for (int n = 0; n < num_nodes; ++n) {
      pos_class_counts[n].assign(
          h_class_counts.begin() + n * 2 * num_classes + 1 * num_classes,
          h_class_counts.begin() + n * 2 * num_classes + 2 * num_classes);
      neg_class_counts[n].assign(
          h_class_counts.begin() + n * 2 * num_classes + 0 * num_classes,
          h_class_counts.begin() + n * 2 * num_classes + 1 * num_classes);
      pos_counts[n] = h_counters[n];
      neg_counts[n] = h_counters[num_nodes + n];
    }

    auto t3 = now();
    // No D2D compaction — kernel wrote directly to d_selected_examples
    auto t4 = t3;

    cudaFree(d_read_buf);
    cudaFree(d_class_counts);
    cudaFree(d_atomic_counters);
    cudaFree(d_child_offsets);

    auto t5 = now();

    auto ms = ms_between;

    s_sad_malloc      += ms(t0, t1);  // includes contiguous D2D + child_offsets upload
    s_sad_flags       += 0;
    s_sad_prefix      += 0;
    s_sad_scatter     += ms(t1, t2);  // direct scatter kernel
    s_sad_d2h_scatter += 0;
    s_sad_d2h_counts  += ms(t2, t3);
    s_sad_d2d_copy    += ms(t3, t4);  // 0 — no D2D
    s_sad_free        += ms(t4, t5);

    return std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer>();
#else
    auto t0 = now();

    int* d_read_buf;
    CUDA_CHECK(cudaMalloc(&d_read_buf, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_read_buf, d_selected_examples,
                          total_rows * sizeof(int), cudaMemcpyDeviceToDevice));

    const int counts_size = num_nodes * 2 * num_classes;
    int* d_class_counts;
    CUDA_CHECK(cudaMalloc(&d_class_counts, counts_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_class_counts, 0, counts_size * sizeof(int)));

    child_offsets.assign(num_nodes, -1);
    int write_offset = 0;
    for (int n = 0; n < num_nodes; ++n) {
      if (!node_has_split[n]) continue;
      child_offsets[n] = write_offset;
      write_offset += node_row_off[n + 1] - node_row_off[n];
    }

    int* d_child_offsets;
    CUDA_CHECK(cudaMalloc(&d_child_offsets, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_child_offsets, child_offsets.data(),
                          num_nodes * sizeof(int), cudaMemcpyHostToDevice));

    int* d_block_right_counts;
    int* d_block_right_prefix;
    CUDA_CHECK(cudaMalloc(&d_block_right_counts, total_node_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_right_prefix, total_node_blocks * sizeof(int)));

    auto t1 = now();

    // Pass 1: count right-going examples per block
    StablePartitionCountKernel<BLOCK><<<total_node_blocks, BLOCK>>>(
        d_col_add_projected, d_best_proj_per_node, d_best_threshold_per_node,
        d_node_row_off, d_node_block_off, d_child_offsets,
        d_block_right_counts, num_proj, num_nodes);

    // Prefix sum over per-block counts
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes,
        d_block_right_counts, d_block_right_prefix, total_node_blocks);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes,
        d_block_right_counts, d_block_right_prefix, total_node_blocks);
    cudaFree(d_temp);

    // Pass 2: stable scatter + class counts
    StablePartitionScatterKernel<BLOCK><<<total_node_blocks, BLOCK>>>(
        d_col_add_projected, d_read_buf, d_selected_examples,
        d_best_proj_per_node, d_best_threshold_per_node,
        d_node_row_off, d_node_block_off, d_child_offsets,
        d_labels,
#ifdef DEDUP_BOOTSTRAP
        d_ex_dup_counts,
#endif
        d_class_counts, num_classes,
        d_block_right_prefix, num_proj, num_nodes);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    auto t2 = now();

    // D2H: class counts
    std::vector<int> h_class_counts(counts_size);
    CUDA_CHECK(cudaMemcpy(h_class_counts.data(), d_class_counts,
                          counts_size * sizeof(int), cudaMemcpyDeviceToHost));

    // D2H: per-block right counts to derive per-node pos/neg totals
    std::vector<int> h_block_right_counts(total_node_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_right_counts.data(), d_block_right_counts,
                          total_node_blocks * sizeof(int), cudaMemcpyDeviceToHost));

    // Compute host-side block offsets from node_row_off
    std::vector<int> h_node_block_off(num_nodes + 1);
    h_node_block_off[0] = 0;
    for (int n = 0; n < num_nodes; ++n) {
      int rows_n = node_row_off[n + 1] - node_row_off[n];
      h_node_block_off[n + 1] = h_node_block_off[n] + (rows_n + BLOCK - 1) / BLOCK;
    }

    pos_class_counts.resize(num_nodes);
    neg_class_counts.resize(num_nodes);
    pos_counts.resize(num_nodes);
    neg_counts.resize(num_nodes);
    for (int n = 0; n < num_nodes; ++n) {
      pos_class_counts[n].assign(
          h_class_counts.begin() + n * 2 * num_classes + 1 * num_classes,
          h_class_counts.begin() + n * 2 * num_classes + 2 * num_classes);
      neg_class_counts[n].assign(
          h_class_counts.begin() + n * 2 * num_classes + 0 * num_classes,
          h_class_counts.begin() + n * 2 * num_classes + 1 * num_classes);
      int right_total = 0;
      int blk_start = h_node_block_off[n];
      int blk_end   = h_node_block_off[n + 1];
      for (int b = blk_start; b < blk_end; ++b)
        right_total += h_block_right_counts[b];
      int rows_node = node_row_off[n + 1] - node_row_off[n];
      pos_counts[n] = right_total;
      neg_counts[n] = rows_node - right_total;
    }

    auto t3 = now();
    auto t4 = t3;

    cudaFree(d_read_buf);
    cudaFree(d_class_counts);
    cudaFree(d_block_right_counts);
    cudaFree(d_block_right_prefix);
    cudaFree(d_child_offsets);

    auto t5 = now();

    auto ms = ms_between;

    s_sad_malloc      += ms(t0, t1);
    s_sad_flags       += 0;
    s_sad_prefix      += 0;
    s_sad_scatter     += ms(t1, t2);
    s_sad_d2h_scatter += 0;
    s_sad_d2h_counts  += ms(t2, t3);
    s_sad_d2d_copy    += ms(t3, t4);
    s_sad_free        += ms(t4, t5);

    return std::vector<yggdrasil_decision_forests::model::decision_tree::ExampleSplitRollingBuffer>();
#endif
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
        const long long base = (long long)row_start * num_proj + (long long)proj_id * rows_node;
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
        long long block_id = (long long)num_proj * node_block_off[node_id] + (long long)proj_id * blocks_node + blockIdx.x;
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
        const long long base = (long long)row_start * num_proj + (long long)proj_id * rows_node;
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
        long long block_id = (long long)num_proj * node_block_off[node_id] + (long long)proj_id * blocks_node + local_bx;
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
        long long gid = (long long)num_proj * node_block_off[node_id] + (long long)proj_id * blocks_node + gid_local;
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
        long long gid = (long long)num_proj * node_block_off[node_id] + (long long)proj_id * blocks_node + gid_local;
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
#ifdef DEDUP_BOOTSTRAP
        const int*    __restrict__ d_ex_dup_counts,  // ADDED: per-row bootstrap multiplicity
#endif
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
        int example_idx = d_selected_examples[row_start + r];  // ADDED: extract for reuse
        int lbl = 2 - d_labels[example_idx];                   // CHANGED: was d_labels[d_selected_examples[row_start + r]]
        int bin = cub::LowerBound(sh_splits, num_bins - 1, val);
#ifdef DEDUP_BOOTSTRAP
        atomicAdd(&sh_hist[lbl * num_bins + bin], d_ex_dup_counts[example_idx]);  // CHANGED: was 1
#else
        atomicAdd(&sh_hist[lbl * num_bins + bin], 1);
#endif
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

// Warp-aggregated variant of BuildHistogramRandomKernel_2D.
// Bit-identical output; only the per-row shared-memory accumulation differs.
// Lanes in a warp that target the same (class,bin) key are collapsed via
// __match_any_sync, so each distinct key costs ONE smem atomicAdd per warp
// instead of up to 32.  The per-group summed weight is obtained with
// __reduce_add_sync (sm_80+), which correctly folds DEDUP_BOOTSTRAP per-row
// multiplicities (not just counts).  The global flush is unchanged.
template<int BLOCK>
__global__ void BuildHistogramRandomKernel_2D_WarpAgg(
        const float*  __restrict__ projected,
        const int*    __restrict__ d_selected_examples,
        const int*    __restrict__ d_labels,
#ifdef DEDUP_BOOTSTRAP
        const int*    __restrict__ d_ex_dup_counts,
#endif
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

    // All lanes reach the ballot (the only prior early-out is uniform per block).
    unsigned active_mask = __ballot_sync(0xFFFFFFFFu, active);
    if (active)
    {
        const int r = gid;
        float val = projected[(size_t)row_start * num_proj +
                               (size_t)proj_id * rows_node + r];
        int example_idx = d_selected_examples[row_start + r];
        int lbl = 2 - d_labels[example_idx];
        int bin = cub::LowerBound(sh_splits, num_bins - 1, val);
        int key = lbl * num_bins + bin;
#ifdef DEDUP_BOOTSTRAP
        int weight = d_ex_dup_counts[example_idx];
#else
        int weight = 1;
#endif
        // Collapse same-key lanes: one smem atomic per distinct key per warp.
        unsigned match = __match_any_sync(active_mask, key);
        int group_sum = __reduce_add_sync(match, weight);   // sm_80+
        int leader = __ffs(match) - 1;
        if ((threadIdx.x & 31) == leader)
            atomicAdd(&sh_hist[key], group_sum);
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

// Launch macro so the DEDUP_BOOTSTRAP arg list is written once.
#ifdef DEDUP_BOOTSTRAP
#define YDF_LAUNCH_HIST(KERN, H0, H1) \
    KERN<BLOCK><<<grid, BLOCK, sharedMemSize>>>( \
        d_col_add_projected, d_selected_examples, d_labels, d_ex_dup_counts, \
        H0, H1, num_proj, num_bins, d_sorted_candidate_splits, \
        d_node_row_off, d_node_block_off, num_nodes)
#else
#define YDF_LAUNCH_HIST(KERN, H0, H1) \
    KERN<BLOCK><<<grid, BLOCK, sharedMemSize>>>( \
        d_col_add_projected, d_selected_examples, d_labels, \
        H0, H1, num_proj, num_bins, d_sorted_candidate_splits, \
        d_node_row_off, d_node_block_off, num_nodes)
#endif

void BuildHistogramRandomKernel_2D_wrap(const float* d_col_add_projected,
                                    const int* d_selected_examples,
                                    const int* d_labels,
#ifdef DEDUP_BOOTSTRAP
                                    const int* d_ex_dup_counts,  // ADDED: per-row bootstrap multiplicity
#endif
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

    // Kernel selection: original is the default; opt in to warp-agg with
    // HIST_WARPAGG=1.  Opt-in A/B timing + correctness check with HIST_AB=1.
    static const bool use_warpagg =
        [] { const char* e = getenv("HIST_WARPAGG"); return e && e[0] == '1'; }();
    static const bool hist_ab =
        [] { const char* e = getenv("HIST_AB"); return e && e[0] == '1'; }();

    // ── A/B: ORIG vs WARPAGG (opt-in, first 10 calls) ──
    static int  hist_bench_call = 0;
    static bool hist_hdr = false;
    if (hist_ab && hist_bench_call < 10) {
        const int hist_elems = num_nodes * num_proj * num_bins;
        int *d_h0 = nullptr, *d_h1 = nullptr, *d_h0b = nullptr, *d_h1b = nullptr;
        CUDA_CHECK(cudaMalloc(&d_h0,  hist_elems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_h1,  hist_elems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_h0b, hist_elems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_h1b, hist_elems * sizeof(int)));
        constexpr int WARMUP = 3, ITERS = 20;
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));

        float orig_ms = 0;
        for (int i = 0; i < WARMUP + ITERS; ++i) {
            CUDA_CHECK(cudaMemset(d_h0, 0, hist_elems * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_h1, 0, hist_elems * sizeof(int)));
            CUDA_CHECK(cudaEventRecord(e0));
            YDF_LAUNCH_HIST(BuildHistogramRandomKernel_2D, d_h0, d_h1);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            if (i >= WARMUP) { float ms; cudaEventElapsedTime(&ms, e0, e1); orig_ms += ms; }
        }

        float wa_ms = 0;
        for (int i = 0; i < WARMUP + ITERS; ++i) {
            CUDA_CHECK(cudaMemset(d_h0b, 0, hist_elems * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_h1b, 0, hist_elems * sizeof(int)));
            CUDA_CHECK(cudaEventRecord(e0));
            YDF_LAUNCH_HIST(BuildHistogramRandomKernel_2D_WarpAgg, d_h0b, d_h1b);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            if (i >= WARMUP) { float ms; cudaEventElapsedTime(&ms, e0, e1); wa_ms += ms; }
        }

        // Correctness: warp-agg must match the original exactly.
        std::vector<int> a(hist_elems), b(hist_elems);
        long mism = 0;
        CUDA_CHECK(cudaMemcpy(a.data(), d_h0,  hist_elems * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_h0b, hist_elems * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < hist_elems; ++i) if (a[i] != b[i]) ++mism;
        CUDA_CHECK(cudaMemcpy(a.data(), d_h1,  hist_elems * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_h1b, hist_elems * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < hist_elems; ++i) if (a[i] != b[i]) ++mism;

        float o = orig_ms / ITERS, w = wa_ms / ITERS;
        if (!hist_hdr) {
            fprintf(stderr,
                "\n══ BuildHistogram A/B: ORIG vs WARPAGG ══\n"
                "  %-6s %-6s %-6s %-11s %-13s %-9s %-10s\n",
                "nodes", "proj", "bins", "ORIG(ms)", "WARPAGG(ms)", "O/W", "match");
            hist_hdr = true;
        }
        fprintf(stderr, "  %-6d %-6d %-6d %-11.4f %-13.4f %-9.2fx %-10s\n",
            num_nodes, num_proj, num_bins, o, w, o / w,
            mism == 0 ? "OK" : "MISMATCH!");

        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        cudaFree(d_h0); cudaFree(d_h1); cudaFree(d_h0b); cudaFree(d_h1b);
        hist_bench_call++;
    }

    // ── actual run ──
    if (use_warpagg) {
        YDF_LAUNCH_HIST(BuildHistogramRandomKernel_2D_WarpAgg, d_hist_class0, d_hist_class1);
    } else {
        YDF_LAUNCH_HIST(BuildHistogramRandomKernel_2D, d_hist_class0, d_hist_class1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
#undef YDF_LAUNCH_HIST



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
    const int node_id = blockIdx.x;
    const int proj_id = blockIdx.y;
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
    dim3 grid(num_nodes, num_proj, 1);
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

    const long long base = (long long)row_start * num_proj + (long long)proj_id * rows_node;
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

    const long long base = (long long)row_start * num_proj + (long long)proj_id * rows_node;
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

    const long long base = (long long)row_start * num_proj + (long long)proj_id * rows_node;
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

    // CUDA-event timing so prior async work is not absorbed into the measured
    // kernel span. cudaEventSynchronize still blocks the host (like the prior
    // cudaDeviceSynchronize), so the caller's external CPU bracket around this
    // wrapper remains valid.
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));
    dim3 grid(total_blocks, num_proj);
    ColumnAddProjectionKernel_SRDW_1_2D<BLOCK><<<grid, BLOCK>>>(
        d_flat_data, d_selected_examples, d_col_add_projected,
        d_offset, d_flat_projection_col_idx, d_flat_projection_weights,
        d_node_row_off, d_node_block_off, num_nodes, num_proj, num_rows, total_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    // float kernel_ms = 0.f; cudaEventElapsedTime(&kernel_ms, e0, e1);
    // std::cerr << "SRDW_1_2D_wrap: Kernel=" << kernel_ms << "ms\n";

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
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

    const long long base = (long long)row_start * num_proj + (long long)proj_id * rows_node;
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

__global__ void ComputeDRSWArrays(
        const int* __restrict__ d_selected_examples,
        const int* __restrict__ node_row_off,
        int* __restrict__ node_ids,
        int* __restrict__ node_offsets,
        int num_nodes,
        int num_total_rows)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_total_rows) return;

    int node_id = cub::UpperBound(node_row_off, num_nodes + 1, pos) - 1;
    int ex_idx = d_selected_examples[pos];
    node_ids[ex_idx] = node_id;
    node_offsets[ex_idx] = pos - node_row_off[node_id];
}

__global__ void ColumnAddProjectionKernel_DRSW_1(
        const float*        __restrict__ dataset,
        float*              __restrict__ projected,
        const int*          __restrict__ d_selected_examples,
        const int*          __restrict__ node_ids,
        const int*          __restrict__ node_offsets,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_proj,
        int                 num_rows,
        int                 num_nodes,
        int                 num_total_rows)
{
    const int proj_id = blockIdx.y;
    const int row_stride = gridDim.x * blockDim.x;
    const int global_row = blockIdx.x * blockDim.x + threadIdx.x;

    for (int r = global_row; r < num_rows; r += row_stride)
    {
        const int node_id = node_ids[r];
        if (node_id < 0 || node_id >= num_nodes) continue;

        const int row_start = node_row_off[node_id];
        const int rows_node = node_row_off[node_id + 1] - row_start;

        const int node_offset = node_offsets[r];

        const int seg_id = node_id * num_proj + proj_id;

        const int begin = col_offset[seg_id];
        const int end   = col_offset[seg_id + 1];

        float sum = 0.0f;
        for (int idx = begin; idx < end; ++idx) {
            int   col   = flat_col_data[idx];
            float w     = flat_weights[idx];
            const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_rows + r;
            const float x = dataset[dataset_idx];
            sum += w * x;
        }
        const long long base = (long long)row_start * num_proj + (long long)proj_id * rows_node;
        projected[base + node_offset] = sum;

    }
}


__global__ void ColumnAddProjectionKernel_DRSW_ActiveRows(
        const float*        __restrict__ dataset,
        float*              __restrict__ projected,
        const int*          __restrict__ active_rows,
        const int*          __restrict__ node_ids,
        const int*          __restrict__ node_offsets,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_proj,
        int                 num_active,
        int                 num_rows,
        int                 num_nodes)
{
    const int proj_id = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active) return;

    const int r = active_rows[idx];
    const int node_id = node_ids[r];
    if (node_id < 0 || node_id >= num_nodes) return;

    const int row_start = node_row_off[node_id];
    const int rows_node = node_row_off[node_id + 1] - row_start;
    const int node_offset = node_offsets[r];

    const int seg_id = node_id * num_proj + proj_id;
    const int begin = col_offset[seg_id];
    const int end   = col_offset[seg_id + 1];

    float sum = 0.0f;
    for (int i = begin; i < end; ++i) {
        int   col = flat_col_data[i];
        float w   = flat_weights[i];
        const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_rows + r;
        sum += w * dataset[dataset_idx];
    }
    const long long base = (long long)row_start * num_proj + (long long)proj_id * rows_node;
    projected[base + node_offset] = sum;
}


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
    int num_total_rows)
{
    constexpr int BLOCK = 256;

    // Time with CUDA events rather than host clocks. Events are enqueued in
    // stream order, so any GPU work queued before this call drains *before* e0
    // fires and is never charged to DRSW setup. (The previous host-clock +
    // cudaDeviceSynchronize scheme absorbed all outstanding async work into
    // s_drsw_compute, which is why drsw_setup could exceed the whole tree time.)
    cudaEvent_t e0, e1, e2;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemset(d_node_ids, -1, static_cast<size_t>(num_rows) * sizeof(int)));

    const int drsw_blocks = (num_total_rows + BLOCK - 1) / BLOCK;
    ComputeDRSWArrays<<<drsw_blocks, BLOCK>>>(
        d_selected_examples, d_node_row_off,
        d_node_ids, d_node_offsets,
        num_nodes, num_total_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e1));  // end of setup (memset + ComputeDRSWArrays)

    const int proj_blocks = (num_rows + BLOCK - 1) / BLOCK;
    dim3 grid(proj_blocks, num_proj);
    ColumnAddProjectionKernel_DRSW_1<<<grid, BLOCK>>>(
        d_flat_data, d_col_add_projected,
        d_selected_examples, d_node_ids, d_node_offsets,
        d_offset, d_flat_projection_col_idx, d_flat_projection_weights,
        d_node_row_off, num_proj, num_rows, num_nodes, num_total_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e2));  // end of projection kernel

    // Single host-side wait, equivalent to the prior final sync.
    CUDA_CHECK(cudaEventSynchronize(e2));

    float setup_ms = 0.f, proj_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&setup_ms, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&proj_ms,  e1, e2));
    s_drsw_compute += setup_ms;
    s_drsw_proj    += proj_ms;

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
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
//         auto start_kernel_DRSW_1 = now();
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
//         auto end_kernel_DRSW_1 = now();
//         std::chrono::duration<double, std::milli> kernel_DRSW_1_duration = end_kernel_DRSW_1 - start_kernel_DRSW_1;
//         std::cout << "Time taken for ColumnAddProjectionKernel_DRSW_1: " << kernel_DRSW_1_duration.count() << " ms" << std::endl;

//     } 

//     else { 

//         dim3 gridAP_SRDW_1(blocks_per_node, num_nodes, num_proj); // 3D grid with node_id as the z-dimension
//         auto start_kernel_SRDW_1 = now();
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
//         auto end_kernel_SRDW_1 = now();
//         std::chrono::duration<double, std::milli> kernel_SRDW_1_duration = end_kernel_SRDW_1 - start_kernel_SRDW_1;
//         std::cout << "Time taken for ColumnAddProjectionKernel_SRDW_1: " << kernel_SRDW_1_duration.count() << " ms" << std::endl;


//         dim3 gridAP_SRDW_2(blocks_per_node, num_proj, num_nodes); // 3D grid with node_id as the z-dimension
//         auto start_kernel_SRDW_2 = now();
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
//         auto end_kernel_SRDW_2 = now();
//         std::chrono::duration<double, std::milli> kernel_SRDW_2_duration = end_kernel_SRDW_2 - start_kernel_SRDW_2;
//         std::cout << "Time taken for ColumnAddProjectionKernel_SRDW_2: " << kernel_SRDW_2_duration.count() << " ms" << std::endl;

//     }
// }

// ═══════════════════════════════════════════════════════════════════════
// NODE_ID_DRSW wrap functions
// ═══════════════════════════════════════════════════════════════════════
#ifdef NODE_ID_DRSW
#include "yggdrasil_decision_forests/learner/decision_tree/node_id_drsw.cuh"

#ifdef NODE_ID_DRSW_SCATTERED_LABEL_HIST
// SRDW-order histogram for the offsets_only path, WITHOUT d_selected_examples.
// Identical in structure to BuildHistogramRandomKernel_2D (one block per node,
// coalesced projected reads, shared-memory histogram, single flush), but the
// per-example label (and dup count) are read coalesced from SRDW-ordered arrays
// d_labels_scattered / d_dup_scattered (filled by the offsets scatter) at row_start + r,
// instead of gathering d_labels[d_selected_examples[row_start + r]].
template<int BLOCK>
__global__ void BuildHistogramSRDWLabelsKernel(
        const float*  __restrict__ projected,
        const int*    __restrict__ d_labels_scattered,
#ifdef DEDUP_BOOTSTRAP
        const int*    __restrict__ d_dup_scattered,
#endif
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
        int lbl = 2 - d_labels_scattered[row_start + r];   // coalesced SRDW read
        int bin = cub::LowerBound(sh_splits, num_bins - 1, val);
#ifdef DEDUP_BOOTSTRAP
        atomicAdd(&sh_hist[lbl * num_bins + bin], d_dup_scattered[row_start + r]);
#else
        atomicAdd(&sh_hist[lbl * num_bins + bin], 1);
#endif
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

void BuildHistogramSRDWLabelsKernel_wrap(
        const float* d_col_add_projected,
        const int*   d_labels_scattered,
#ifdef DEDUP_BOOTSTRAP
        const int*   d_dup_scattered,
#endif
        int*         d_hist_class0,
        int*         d_hist_class1,
        int          num_proj,
        int          num_bins,
        const float* d_sorted_candidate_splits,
        const int*   d_node_row_off,
        const int*   d_node_block_off,
        int          num_nodes,
        int          total_node_blocks)
{
    constexpr int BLOCK = 256;
    int sharedMemSize = 2 * num_bins * sizeof(int) +
                        (num_bins - 1) * sizeof(float);
    dim3 grid(total_node_blocks, num_proj);
    BuildHistogramSRDWLabelsKernel<BLOCK><<<grid, BLOCK, sharedMemSize>>>(
        d_col_add_projected,
        d_labels_scattered,
#ifdef DEDUP_BOOTSTRAP
        d_dup_scattered,
#endif
        d_hist_class0,
        d_hist_class1,
        num_proj,
        num_bins,
        d_sorted_candidate_splits,
        d_node_row_off,
        d_node_block_off,
        num_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif  // NODE_ID_DRSW_SCATTERED_LABEL_HIST

void NodeIdDRSW_Init_wrap(
    const int* d_selected_examples,
    const int* d_node_row_off,
    int* d_node_ids,
    int num_rows,
    int num_nodes,
    int num_total_selected)
{
    NodeIdDRSW_Init(d_selected_examples, d_node_row_off,
                    d_node_ids, num_rows, num_nodes, num_total_selected);
}

void NodeIdDRSW_ComputeOffsets_wrap(
    int* d_node_offsets,
    int* d_selected_examples,
    const int* d_node_ids,
    const int* d_node_row_off,
    int num_rows,
    int num_nodes,
    int total_rows)
{
    NodeIdDRSW_ComputeOffsets(d_node_offsets, d_selected_examples,
                              d_node_ids, d_node_row_off,
                              num_rows, num_nodes, total_rows);
}

void NodeIdDRSW_ComputeOffsetsOnly_wrap(
    int* d_node_offsets,
    const int* d_node_ids,
    const int* d_node_row_off,
    int num_rows,
    int num_nodes
#ifdef NODE_ID_DRSW_SCATTERED_LABEL_HIST
    ,
    const int* d_labels,
    const int* d_dup,
    int* d_labels_scattered,
    int* d_dup_scattered
#endif
    )
{
    NodeIdDRSW_ComputeOffsetsOnly(d_node_offsets, d_node_ids,
                                   d_node_row_off,
                                   num_rows, num_nodes
#ifdef NODE_ID_DRSW_SCATTERED_LABEL_HIST
                                   , d_labels, d_dup, d_labels_scattered, d_dup_scattered
#endif
                                   );
}

void NodeIdDRSW_ProjectionOnly_wrap(
    const float* d_flat_data,
    float* d_col_add_projected,
    const int* d_node_ids,
    const int* d_node_offsets,
    const int* d_offset,
    const int* d_flat_projection_col_idx,
    const float* d_flat_projection_weights,
    const int* d_node_row_off,
    int num_proj,
    int num_rows,
    int num_nodes)
{
    constexpr int BLOCK = 256;
    const int proj_blocks = (num_rows + BLOCK - 1) / BLOCK;
    dim3 grid(proj_blocks, num_proj);
    ColumnAddProjectionKernel_DRSW_1<<<grid, BLOCK>>>(
        d_flat_data, d_col_add_projected,
        nullptr, d_node_ids, d_node_offsets,
        d_offset, d_flat_projection_col_idx, d_flat_projection_weights,
        d_node_row_off, num_proj, num_rows, num_nodes, num_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

NodeIdDRSW_SplitResult_Host NodeIdDRSW_SplitUpdate_wrap(
    int* d_node_ids,
    const int* d_node_offsets,
    const float* d_projected,
    const int* d_best_proj_per_node,
    const float* d_best_threshold_per_node,
    const int* d_node_row_off,
    const int* d_labels,
#ifdef DEDUP_BOOTSTRAP
    const int* d_dup_counts,
#endif
    const std::vector<bool>& node_has_split,
    const std::vector<int>& node_counts,
    const std::vector<int>& node_row_offsets,
    int num_rows,
    int num_active_nodes,
    int num_proj,
    int num_classes)
{
    auto result = NodeIdDRSW_SplitUpdate(
        d_node_ids, d_node_offsets, d_projected,
        d_best_proj_per_node, d_best_threshold_per_node,
        d_node_row_off,
        d_labels,
#ifdef DEDUP_BOOTSTRAP
        d_dup_counts,
#endif
        node_has_split, node_counts, node_row_offsets,
        num_rows, num_active_nodes, num_proj, num_classes);

    NodeIdDRSW_SplitResult_Host host_result;
    host_result.num_children = result.num_children;
    host_result.pos_counts = std::move(result.pos_counts);
    host_result.neg_counts = std::move(result.neg_counts);
    host_result.pos_class_counts = std::move(result.pos_class_counts);
    host_result.neg_class_counts = std::move(result.neg_class_counts);
    host_result.child_offsets = std::move(result.child_offsets);
    return host_result;
}

#endif  // NODE_ID_DRSW
