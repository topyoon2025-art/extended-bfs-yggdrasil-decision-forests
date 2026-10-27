// Node-ID-Only DRSW  (NODE_ID_DRSW)
//
// Single persistent array: d_node_ids[num_rows].
// Per-depth: ComputeOffsets derives d_node_offsets + rebuilds d_selected_examples
// via per-node atomic counters — replaces both CUB sort and ComputeDRSWArrays.
//
// Projection uses the existing DRSW kernel (iterates all num_rows, uses
// node_ids + node_offsets to write SRDW layout). MinMax and Histogram
// kernels are reused unchanged (d_selected_examples rebuilt by ComputeOffsets).
//
// SplitUpdate iterates all num_rows, reads node_ids[row] directly (no
// CUB UpperBound), rewrites node_ids parent→child, accumulates counts.

#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <iostream>
#include <limits>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)
#endif


// ═══════════════════════════════════════════════════════════════════════
// 1. INIT — set node_ids from bootstrap selected_examples
// ═══════════════════════════════════════════════════════════════════════

__global__ void NodeIdDRSW_InitKernel(
        const int* __restrict__ d_selected_examples,
        const int* __restrict__ node_row_off,
        int* __restrict__ node_ids,
        int num_nodes,
        int num_total_selected)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_total_selected) return;

    int node_id = cub::UpperBound(node_row_off, num_nodes + 1, pos) - 1;
    node_ids[d_selected_examples[pos]] = node_id;
}

inline void NodeIdDRSW_Init(
        const int* d_selected_examples,
        const int* d_node_row_off,
        int* d_node_ids,
        int num_rows,
        int num_nodes,
        int num_total_selected)
{
    constexpr int BLOCK = 256;
    CUDA_CHECK(cudaMemset(d_node_ids, -1,
               static_cast<size_t>(num_rows) * sizeof(int)));
    int blocks = (num_total_selected + BLOCK - 1) / BLOCK;
    NodeIdDRSW_InitKernel<<<blocks, BLOCK>>>(
        d_selected_examples, d_node_row_off,
        d_node_ids, num_nodes, num_total_selected);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ═══════════════════════════════════════════════════════════════════════
// 2. COMPUTE OFFSETS — derive node_offsets + rebuild d_selected_examples
//    One pass over num_rows with per-node atomic counters.
//    Replaces both CUB sort and ComputeDRSWArrays.
// ═══════════════════════════════════════════════════════════════════════

// Max nodes per depth for the shared-memory ranking kernels below (one
// s_counts / s_base slot per node). The only ComputeOffsets(Only) kernels.
constexpr int kMaxOffsetNodes = 256;

// Per-thread shared-memory atomicAdd for ranking, for num_nodes <= SMEM_NODES.
template <int SMEM_NODES>
__global__ void NodeIdDRSW_ComputeOffsetsKernel_SmemAtomicAdd(
        const int* __restrict__ node_ids,
        int*       __restrict__ node_offsets,
        int*       __restrict__ d_selected_examples,
        const int* __restrict__ node_row_off,
        int*       __restrict__ d_atomic_counters,
        int num_rows,
        int num_nodes)
{
    __shared__ int s_counts[SMEM_NODES];
    __shared__ int s_base[SMEM_NODES];

    for (int i = threadIdx.x; i < num_nodes; i += blockDim.x)
        s_counts[i] = 0;
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int nid = -1;
    if (row < num_rows) {
        nid = node_ids[row];
        if (nid < 0 || nid >= num_nodes) nid = -1;
    }

    int local_offset = -1;
    if (nid >= 0)
        local_offset = atomicAdd(&s_counts[nid], 1);
    __syncthreads();

    if (threadIdx.x < num_nodes && s_counts[threadIdx.x] > 0)
        s_base[threadIdx.x] = atomicAdd(&d_atomic_counters[threadIdx.x],
                                         s_counts[threadIdx.x]);
    __syncthreads();

    if (nid < 0) return;
    int offset = s_base[nid] + local_offset;
    node_offsets[row] = offset;
    d_selected_examples[node_row_off[nid] + offset] = row;
}

inline void NodeIdDRSW_ComputeOffsets(
        int* d_node_offsets,
        int* d_selected_examples,
        const int* d_node_ids,
        const int* d_node_row_off,
        int num_rows,
        int num_nodes,
        int total_rows)
{
    constexpr int BLOCK = 256;

    static int* d_atomic_counters = nullptr;
    static int counters_capacity = 0;
    if (num_nodes > counters_capacity) {
        if (d_atomic_counters) cudaFree(d_atomic_counters);
        int alloc_size = std::max(num_nodes, 65536);
        CUDA_CHECK(cudaMalloc(&d_atomic_counters, alloc_size * sizeof(int)));
        counters_capacity = alloc_size;
    }

    int blocks = (num_rows + BLOCK - 1) / BLOCK;

    // Only the shared-memory ranking kernel exists (one smem counter slot per
    // node), so at most kMaxOffsetNodes nodes per depth: tree_depth <= 10, since
    // the last depth only makes leaves. The >256-node global-atomic fallback was
    // removed; fail loudly instead.
    if (num_nodes > kMaxOffsetNodes) {
        fprintf(stderr, "%s: %d nodes at this depth exceeds the %d-node limit "
                "(the >256-node fallback kernel was removed; use tree_depth <= 10)\n",
                __func__, num_nodes, kMaxOffsetNodes);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
    NodeIdDRSW_ComputeOffsetsKernel_SmemAtomicAdd<kMaxOffsetNodes><<<blocks, BLOCK>>>(
        d_node_ids, d_node_offsets, d_selected_examples,
        d_node_row_off, d_atomic_counters, num_rows, num_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Shared-memory-ranked offsets-only for num_nodes <= SMEM_NODES.  Mirrors
// NodeIdDRSW_ComputeOffsetsKernel_SmemAtomicAdd (cheap intra-block ranking in
// shared memory + one global atomicAdd per node per block), but skips the
// d_selected_examples store (except under NODE_ID_SRDW_PROJ).
// WRITE_LABELS additionally scatters labels/dups into SRDW layout at the slot.
template <int SMEM_NODES, bool WRITE_LABELS>
__global__ void NodeIdDRSW_ComputeOffsetsOnlyKernel_Smem(
        const int* __restrict__ node_ids,
        int*       __restrict__ node_offsets,
        const int* __restrict__ node_row_off,
        int*       __restrict__ d_atomic_counters,
        int num_rows,
        int num_nodes,
        const int* __restrict__ d_labels = nullptr,
        const int* __restrict__ d_dup = nullptr,
        int*       __restrict__ d_labels_scattered = nullptr,
        int*       __restrict__ d_dup_scattered = nullptr,
        int*       __restrict__ d_selected_examples = nullptr)
{
    __shared__ int s_counts[SMEM_NODES];
    __shared__ int s_base[SMEM_NODES];

    for (int i = threadIdx.x; i < num_nodes; i += blockDim.x)
        s_counts[i] = 0;
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int nid = -1;
    if (row < num_rows) {
        nid = node_ids[row];
        if (nid < 0 || nid >= num_nodes) nid = -1;
    }

    int local_offset = -1;
    if (nid >= 0)
        local_offset = atomicAdd(&s_counts[nid], 1);
    __syncthreads();

    if (threadIdx.x < num_nodes && s_counts[threadIdx.x] > 0)
        s_base[threadIdx.x] = atomicAdd(&d_atomic_counters[threadIdx.x],
                                         s_counts[threadIdx.x]);
    __syncthreads();

    if (nid < 0) return;
    int offset = s_base[nid] + local_offset;
    node_offsets[row] = offset;
    if (WRITE_LABELS) {
        const int slot = node_row_off[nid] + offset;
        d_labels_scattered[slot] = d_labels[row]; // is the class label value. 
#ifdef DEDUP_BOOTSTRAP
        d_dup_scattered[slot] = d_dup[row]; // count
#endif
    }
#ifdef NODE_ID_SRDW_PROJ
    // Sorted row list for the SRDW_1_2D projection kernel.
    d_selected_examples[node_row_off[nid] + offset] = row;
#endif
}

// The d_labels/d_dup/d_labels_scattered/d_dup_scattered pointers are used only in the
// NODE_ID_DRSW_SCATTERED_LABEL_HIST build: there the scatter also writes the per-example
// label (and dup count) into SRDW layout so the SRDW histogram can read them
// coalesced without d_selected_examples. They default to null and are ignored in
// every other build.
// d_selected_examples is written only under NODE_ID_SRDW_PROJ (SRDW_1_2D
// projection needs the node-sorted row list); ignored otherwise.
inline void NodeIdDRSW_ComputeOffsetsOnly(
        int* d_node_offsets,
        const int* d_node_ids,
        const int* d_node_row_off,
        int num_rows,
        int num_nodes,
        const int* d_labels = nullptr,
        const int* d_dup = nullptr,
        int* d_labels_scattered = nullptr,
        int* d_dup_scattered = nullptr,
        int* d_selected_examples = nullptr)
{
    constexpr int BLOCK = 256;
    int blocks = (num_rows + BLOCK - 1) / BLOCK;

#ifdef NODE_ID_DRSW_SCATTERED_LABEL_HIST
    constexpr bool kWriteLabels = true;
#else
    constexpr bool kWriteLabels = false;
#endif

    static int* d_atomic_counters = nullptr;
    static int counters_capacity = 0;
    if (num_nodes > counters_capacity) {
        if (d_atomic_counters) cudaFree(d_atomic_counters);
        int alloc_size = std::max(num_nodes, 65536);
        CUDA_CHECK(cudaMalloc(&d_atomic_counters, alloc_size * sizeof(int)));
        counters_capacity = alloc_size;
    }
    // Only the shared-memory ranking kernel exists (one smem counter slot per
    // node), so at most kMaxOffsetNodes nodes per depth: tree_depth <= 10, since
    // the last depth only makes leaves. The >256-node global-atomic fallback was
    // removed; fail loudly instead.
    if (num_nodes > kMaxOffsetNodes) {
        fprintf(stderr, "%s: %d nodes at this depth exceeds the %d-node limit "
                "(the >256-node fallback kernel was removed; use tree_depth <= 10)\n",
                __func__, num_nodes, kMaxOffsetNodes);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
    NodeIdDRSW_ComputeOffsetsOnlyKernel_Smem<kMaxOffsetNodes, kWriteLabels><<<blocks, BLOCK>>>(
        d_node_ids, d_node_offsets, d_node_row_off, d_atomic_counters,
        num_rows, num_nodes,
        d_labels, d_dup, d_labels_scattered, d_dup_scattered,
        d_selected_examples);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ═══════════════════════════════════════════════════════════════════════
// 3. SPLIT UPDATE — update node_ids, accumulate child counts
//    Iterates all num_rows, reads node_ids[row] directly (no UpperBound).
//    Uses node_offsets[row] to look up projected values in SRDW layout.
// ═══════════════════════════════════════════════════════════════════════

// OLD: per-thread global atomicAdd (kept for A/B comparison)
__global__ void NodeIdDRSW_SplitUpdateKernel_GlobalAtomic(
        int*         __restrict__ node_ids,
        const int*   __restrict__ node_offsets,
        const float* __restrict__ projected,
        const int*   __restrict__ d_best_proj,
        const float* __restrict__ d_best_threshold,
        const int*   __restrict__ d_node_to_neg_child,
        const int*   __restrict__ d_node_to_pos_child,
        const int*   __restrict__ node_row_off,
        const int*   __restrict__ d_labels,
#ifdef DEDUP_BOOTSTRAP
        const int*   __restrict__ d_dup_counts,
#endif
        int*         __restrict__ d_child_counters,
        int*         __restrict__ d_child_class_counts,
        int          num_rows,
        int          num_active_nodes,
        int          num_proj,
        int          num_classes)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int child_id = -1;
    int cls = 0;
    int weight = 0;

    if (row < num_rows) {
        int nid = node_ids[row];
        if (nid >= 0 && nid < num_active_nodes) {
            int neg_child = d_node_to_neg_child[nid];
            int pos_child = d_node_to_pos_child[nid];

            if (neg_child < 0 && pos_child < 0) {
                node_ids[row] = -1;
            } else {
                int best_proj = d_best_proj[nid];
                int row_start = node_row_off[nid];
                int rows_node = node_row_off[nid + 1] - row_start;
                int local_r = node_offsets[row];
                long long base = (long long)row_start * num_proj
                               + (long long)best_proj * rows_node;
                float val = projected[base + local_r];

                child_id = (val >= d_best_threshold[nid]) ? pos_child : neg_child;
                node_ids[row] = child_id;

                cls = d_labels[row];
#ifdef DEDUP_BOOTSTRAP
                weight = d_dup_counts[row];
#else
                weight = 1;
#endif
            }
        }
    }

    if (child_id < 0) return;

    atomicAdd(&d_child_counters[child_id], 1);
    atomicAdd(&d_child_class_counts[child_id * num_classes + cls], weight);
}

// NEW: shared-memory block reduction — accumulate in smem, flush once per block
template <int SMEM_CHILDREN>
__global__ void NodeIdDRSW_SplitUpdateKernel_SmemReduce(
        int*         __restrict__ node_ids,
        const int*   __restrict__ node_offsets,
        const float* __restrict__ projected,
        const int*   __restrict__ d_best_proj,
        const float* __restrict__ d_best_threshold,
        const int*   __restrict__ d_node_to_neg_child,
        const int*   __restrict__ d_node_to_pos_child,
        const int*   __restrict__ node_row_off,
        const int*   __restrict__ d_labels,
#ifdef DEDUP_BOOTSTRAP
        const int*   __restrict__ d_dup_counts,
#endif
        int*         __restrict__ d_child_counters,
        int*         __restrict__ d_child_class_counts,
        int          num_rows,
        int          num_active_nodes,
        int          num_proj,
        int          num_classes,
        int          num_children)
{
    extern __shared__ int smem[];
    int* s_counters    = smem;                        // [num_children]
    int* s_class_counts = smem + num_children;        // [num_children * num_classes]
    const int cc_size = num_children * num_classes;

    for (int i = threadIdx.x; i < num_children + cc_size; i += blockDim.x)
        smem[i] = 0;
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        int nid = node_ids[row];
        if (nid >= 0 && nid < num_active_nodes) {
            int neg_child = d_node_to_neg_child[nid];
            int pos_child = d_node_to_pos_child[nid];

            if (neg_child < 0 && pos_child < 0) {
                node_ids[row] = -1;
            } else {
                int best_proj = d_best_proj[nid];
                int row_start = node_row_off[nid];
                int rows_node = node_row_off[nid + 1] - row_start;
                int local_r = node_offsets[row];
                long long base = (long long)row_start * num_proj
                               + (long long)best_proj * rows_node;
                float val = projected[base + local_r];

                int child_id = (val >= d_best_threshold[nid]) ? pos_child : neg_child;
                node_ids[row] = child_id;

                int cls = d_labels[row];
#ifdef DEDUP_BOOTSTRAP
                int weight = d_dup_counts[row];
#else
                int weight = 1;
#endif
                atomicAdd(&s_counters[child_id], 1);
                atomicAdd(&s_class_counts[child_id * num_classes + cls], weight);
            }
        }
    }

    __syncthreads();

    // Flush shared memory to global with one atomicAdd per key per block
    for (int i = threadIdx.x; i < num_children; i += blockDim.x) {
        if (s_counters[i] != 0)
            atomicAdd(&d_child_counters[i], s_counters[i]);
    }
    for (int i = threadIdx.x; i < cc_size; i += blockDim.x) {
        if (s_class_counts[i] != 0)
            atomicAdd(&d_child_class_counts[i], s_class_counts[i]);
    }
}


struct NodeIdDRSW_SplitResult {
    int num_children;
    std::vector<int> pos_counts;
    std::vector<int> neg_counts;
    std::vector<std::vector<int>> pos_class_counts;
    std::vector<std::vector<int>> neg_class_counts;
    std::vector<int> child_offsets;
    std::vector<int> parent_to_neg_child;
    std::vector<int> parent_to_pos_child;
};

inline NodeIdDRSW_SplitResult NodeIdDRSW_SplitUpdate(
        int*         d_node_ids,
        const int*   d_node_offsets,
        const float* d_projected,
        const int*   d_best_proj_per_node,
        const float* d_best_threshold_per_node,
        const int*   d_node_row_off,
        const int*   d_labels,
#ifdef DEDUP_BOOTSTRAP
        const int*   d_dup_counts,
#endif
        const std::vector<bool>& node_has_split,
        const std::vector<int>&  node_counts,
        const std::vector<int>&  node_row_offsets,
        int num_rows,
        int num_active_nodes,
        int num_proj,
        int num_classes)
{
    constexpr int BLOCK = 256;
    NodeIdDRSW_SplitResult result;

    result.parent_to_neg_child.assign(num_active_nodes, -1);
    result.parent_to_pos_child.assign(num_active_nodes, -1);

    int next_child_id = 0;
    for (int n = 0; n < num_active_nodes; ++n) {
        if (!node_has_split[n]) continue;
        result.parent_to_neg_child[n] = next_child_id++;
        result.parent_to_pos_child[n] = next_child_id++;
    }
    result.num_children = next_child_id;

    if (result.num_children == 0) {
        result.pos_counts.assign(num_active_nodes, 0);
        result.neg_counts.assign(num_active_nodes, 0);
        result.pos_class_counts.resize(num_active_nodes);
        result.neg_class_counts.resize(num_active_nodes);
        for (int n = 0; n < num_active_nodes; ++n) {
            result.pos_class_counts[n].assign(num_classes, 0);
            result.neg_class_counts[n].assign(num_classes, 0);
        }
        result.child_offsets.assign(num_active_nodes, -1);
        return result;
    }

    static int* d_node_to_neg = nullptr;
    static int* d_node_to_pos = nullptr;
    static int split_node_cap = 0;
    if (num_active_nodes > split_node_cap) {
        if (d_node_to_neg) { cudaFree(d_node_to_neg); cudaFree(d_node_to_pos); }
        int alloc = std::max(num_active_nodes, 65536);
        CUDA_CHECK(cudaMalloc(&d_node_to_neg, alloc * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_node_to_pos, alloc * sizeof(int)));
        split_node_cap = alloc;
    }
    CUDA_CHECK(cudaMemcpy(d_node_to_neg, result.parent_to_neg_child.data(),
               num_active_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_to_pos, result.parent_to_pos_child.data(),
               num_active_nodes * sizeof(int), cudaMemcpyHostToDevice));

    static int* d_child_counters = nullptr;
    static int child_cap = 0;
    if (result.num_children > child_cap) {
        if (d_child_counters) cudaFree(d_child_counters);
        int alloc = std::max(result.num_children, 65536);
        CUDA_CHECK(cudaMalloc(&d_child_counters, alloc * sizeof(int)));
        child_cap = alloc;
    }
    CUDA_CHECK(cudaMemset(d_child_counters, 0, result.num_children * sizeof(int)));

    static int* d_child_class_counts = nullptr;
    static int cc_cap = 0;
    const int cc_size = result.num_children * num_classes;
    if (cc_size > cc_cap) {
        if (d_child_class_counts) cudaFree(d_child_class_counts);
        int alloc = std::max(cc_size, 65536 * num_classes);
        CUDA_CHECK(cudaMalloc(&d_child_class_counts, alloc * sizeof(int)));
        cc_cap = alloc;
    }
    CUDA_CHECK(cudaMemset(d_child_class_counts, 0, cc_size * sizeof(int)));

    int blocks = (num_rows + BLOCK - 1) / BLOCK;

    // ── A/B benchmark: GLOBAL atomic vs SMEM reduction (children + class counts) ──
    // Kernel-only timing: node_ids is restored from a saved copy OUTSIDE the
    // timed region each iteration (both kernels rewrite it parent→child), and all
    // outputs go to scratch buffers so the real result below is untouched.
    {
        static int su_bench_call = 0;
        static bool su_header = false;
        // Opt-in only: the A/B loop runs ~40 extra kernel launches per depth and
        // would otherwise inflate the timed SplitUpdate.  Enable with
        // SPLITUPDATE_AB=1 in the environment.
        static const bool su_bench_enabled =
            [] { const char* e = getenv("SPLITUPDATE_AB"); return e && e[0] == '1'; }();
        const size_t bench_smem =
            static_cast<size_t>(result.num_children) * (1 + num_classes) * sizeof(int);

        static int bench_smem_optin = -1;
        if (bench_smem_optin < 0) {
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            CUDA_CHECK(cudaDeviceGetAttribute(
                &bench_smem_optin,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        }

        if (su_bench_enabled && su_bench_call < 10 && result.num_children > 0) {
            constexpr int WARMUP = 3;
            constexpr int ITERS  = 20;

            int* d_ids_orig = nullptr;   // pristine node_ids snapshot
            int* d_ids_work = nullptr;   // mutated by each launch
            int* d_off_scr  = nullptr;   // scratch node_offsets output
            int* d_cnt_scr  = nullptr;
            int* d_cc_scr   = nullptr;
            CUDA_CHECK(cudaMalloc(&d_ids_orig, num_rows * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_ids_work, num_rows * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_off_scr,  num_rows * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cnt_scr,  result.num_children * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cc_scr,   cc_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_ids_orig, d_node_ids,
                       num_rows * sizeof(int), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_off_scr, d_node_offsets,
                       num_rows * sizeof(int), cudaMemcpyDeviceToDevice));

            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0));
            CUDA_CHECK(cudaEventCreate(&e1));

            // --- GLOBAL atomic ---
            float global_ms = 0;
            for (int i = 0; i < WARMUP + ITERS; ++i) {
                CUDA_CHECK(cudaMemcpy(d_ids_work, d_ids_orig,
                           num_rows * sizeof(int), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemset(d_cnt_scr, 0, result.num_children * sizeof(int)));
                CUDA_CHECK(cudaMemset(d_cc_scr, 0, cc_size * sizeof(int)));
                CUDA_CHECK(cudaEventRecord(e0));
                NodeIdDRSW_SplitUpdateKernel_GlobalAtomic<<<blocks, BLOCK>>>(
                    d_ids_work, d_off_scr, d_projected,
                    d_best_proj_per_node, d_best_threshold_per_node,
                    d_node_to_neg, d_node_to_pos, d_node_row_off, d_labels,
#ifdef DEDUP_BOOTSTRAP
                    d_dup_counts,
#endif
                    d_cnt_scr, d_cc_scr,
                    num_rows, num_active_nodes, num_proj, num_classes);
                CUDA_CHECK(cudaEventRecord(e1));
                CUDA_CHECK(cudaEventSynchronize(e1));
                if (i >= WARMUP) {
                    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                    global_ms += ms;
                }
            }

            // --- SMEM reduction (only if it fits) ---
            float smem_ms = -1;
            if (bench_smem <= static_cast<size_t>(bench_smem_optin)) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    NodeIdDRSW_SplitUpdateKernel_SmemReduce<512>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(bench_smem)));
                smem_ms = 0;
                for (int i = 0; i < WARMUP + ITERS; ++i) {
                    CUDA_CHECK(cudaMemcpy(d_ids_work, d_ids_orig,
                               num_rows * sizeof(int), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemset(d_cnt_scr, 0, result.num_children * sizeof(int)));
                    CUDA_CHECK(cudaMemset(d_cc_scr, 0, cc_size * sizeof(int)));
                    CUDA_CHECK(cudaEventRecord(e0));
                    NodeIdDRSW_SplitUpdateKernel_SmemReduce<512>
                        <<<blocks, BLOCK, bench_smem>>>(
                        d_ids_work, d_off_scr, d_projected,
                        d_best_proj_per_node, d_best_threshold_per_node,
                        d_node_to_neg, d_node_to_pos, d_node_row_off, d_labels,
#ifdef DEDUP_BOOTSTRAP
                        d_dup_counts,
#endif
                        d_cnt_scr, d_cc_scr,
                        num_rows, num_active_nodes, num_proj, num_classes,
                        result.num_children);
                    CUDA_CHECK(cudaEventRecord(e1));
                    CUDA_CHECK(cudaEventSynchronize(e1));
                    if (i >= WARMUP) {
                        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                        smem_ms += ms;
                    }
                }
            }

            float g_avg = global_ms / ITERS;
            float s_avg = smem_ms / ITERS;
            if (!su_header) {
                fprintf(stderr,
                    "\n══ SplitUpdate A/B: GLOBAL(atomic) vs SMEM(reduce) "
                    "— children+class counts ══\n"
                    "  %-6s %-6s %-8s %-11s %-11s %-9s\n",
                    "child", "cls", "rows", "GLOBAL(ms)", "SMEM(ms)", "G/S");
                su_header = true;
            }
            if (smem_ms < 0)
                fprintf(stderr, "  %-6d %-6d %-8d %-11.4f %-11s %-9s  (smem %zuB > cap)\n",
                    result.num_children, num_classes, num_rows, g_avg,
                    "n/a", "fallback", bench_smem);
            else
                fprintf(stderr, "  %-6d %-6d %-8d %-11.4f %-11.4f %-9.2fx\n",
                    result.num_children, num_classes, num_rows, g_avg, s_avg,
                    g_avg / s_avg);

            CUDA_CHECK(cudaEventDestroy(e0));
            CUDA_CHECK(cudaEventDestroy(e1));
            CUDA_CHECK(cudaFree(d_ids_orig));
            CUDA_CHECK(cudaFree(d_ids_work));
            CUDA_CHECK(cudaFree(d_off_scr));
            CUDA_CHECK(cudaFree(d_cnt_scr));
            CUDA_CHECK(cudaFree(d_cc_scr));
            su_bench_call++;
        }
    }

    // Toggle (SPLIT_SMEM, --config, default on): prefer shared-memory block
    // reduction. --config=split_global_atomic leaves SPLIT_SMEM undefined and
    // forces the per-row global atomicAdd kernel.
#if defined(SPLIT_SMEM)
    constexpr bool kSplitPreferSmem = true;
#else
    constexpr bool kSplitPreferSmem = false;
#endif

    // The smem-reduction kernel needs (num_children + num_children*num_classes)
    // ints of dynamic shared memory.  That grows with both tree width and class
    // count and can exceed the per-block smem limit on wide/high-class problems,
    // so query the device cap once and fall back to the global-atomic kernel
    // whenever the request would not fit.
    const size_t smem_bytes =
        static_cast<size_t>(result.num_children) * (1 + num_classes) * sizeof(int);

    static int smem_optin_max = -1;
    if (smem_optin_max < 0) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(
            &smem_optin_max,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    }

    bool use_smem = kSplitPreferSmem &&
                    (smem_bytes <= static_cast<size_t>(smem_optin_max));

    if (use_smem) {
        // Opt in to the full dynamic smem budget (needed above the 48 KB
        // default cap).  Done once per observed size; cheap to repeat.
        static size_t smem_optin_set = 0;
        if (smem_bytes > smem_optin_set) {
            CUDA_CHECK(cudaFuncSetAttribute(
                NodeIdDRSW_SplitUpdateKernel_SmemReduce<512>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            smem_optin_set = smem_bytes;
        }
        NodeIdDRSW_SplitUpdateKernel_SmemReduce<512><<<blocks, BLOCK, smem_bytes>>>(
            d_node_ids, d_node_offsets, d_projected,
            d_best_proj_per_node, d_best_threshold_per_node,
            d_node_to_neg, d_node_to_pos,
            d_node_row_off,
            d_labels,
#ifdef DEDUP_BOOTSTRAP
            d_dup_counts,
#endif
            d_child_counters,
            d_child_class_counts,
            num_rows, num_active_nodes, num_proj, num_classes,
            result.num_children);
    } else {
        // Fallback: global atomics (no shared-memory footprint).  Used when
        // --config=split_global_atomic, or when the smem request exceeds the cap.
        NodeIdDRSW_SplitUpdateKernel_GlobalAtomic<<<blocks, BLOCK>>>(
            d_node_ids, d_node_offsets, d_projected,
            d_best_proj_per_node, d_best_threshold_per_node,
            d_node_to_neg, d_node_to_pos,
            d_node_row_off,
            d_labels,
#ifdef DEDUP_BOOTSTRAP
            d_dup_counts,
#endif
            d_child_counters,
            d_child_class_counts,
            num_rows, num_active_nodes, num_proj, num_classes);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_child_counters(result.num_children);
    CUDA_CHECK(cudaMemcpy(h_child_counters.data(), d_child_counters,
               result.num_children * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_class_counts(cc_size);
    CUDA_CHECK(cudaMemcpy(h_class_counts.data(), d_child_class_counts,
               cc_size * sizeof(int), cudaMemcpyDeviceToHost));

    result.pos_counts.resize(num_active_nodes);
    result.neg_counts.resize(num_active_nodes);
    result.pos_class_counts.resize(num_active_nodes);
    result.neg_class_counts.resize(num_active_nodes);
    result.child_offsets.assign(num_active_nodes, -1);

    int write_offset = 0;
    for (int n = 0; n < num_active_nodes; ++n) {
        if (!node_has_split[n]) {
            result.pos_counts[n] = 0;
            result.neg_counts[n] = 0;
            result.pos_class_counts[n].assign(num_classes, 0);
            result.neg_class_counts[n].assign(num_classes, 0);
            continue;
        }

        int neg_id = result.parent_to_neg_child[n];
        int pos_id = result.parent_to_pos_child[n];

        result.neg_counts[n] = h_child_counters[neg_id];
        result.pos_counts[n] = h_child_counters[pos_id];

        result.neg_class_counts[n].assign(
            h_class_counts.begin() + neg_id * num_classes,
            h_class_counts.begin() + (neg_id + 1) * num_classes);
        result.pos_class_counts[n].assign(
            h_class_counts.begin() + pos_id * num_classes,
            h_class_counts.begin() + (pos_id + 1) * num_classes);

        result.child_offsets[n] = write_offset;
        write_offset += node_counts[n];
    }

    return result;
}


