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

// Forward declaration
inline bool NodeIdDRSW_VerifySorted(
        const int* d_selected_examples, const int* d_node_row_off,
        int num_nodes);

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

// OLD kernel: per-thread shared-memory atomicAdd for ranking.
// Kept for A/B timing comparison against the warp-ranked version.
template <int SMEM_NODES>
__global__ void NodeIdDRSW_ComputeOffsetsKernel_Smem_OldAtomicAdd(
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

// Block-reservation kernel with deterministic warp-based ranking.
// Each warp counts its per-node elements via __match_any_sync, then a
// shared-memory prefix sum across warps assigns ranks in warp order:
// warp 0's rows first, then warp 1's, etc.  Within each warp, lower lane
// (= lower row index) gets the lower rank.  Result: within each block's
// contribution to a node, elements in d_selected_examples are sorted by
// row index.  Inter-block order depends on global-atomic reservation.
// Shared-memory path: num_nodes <= SMEM_NODES
template <int SMEM_NODES>
__global__ void NodeIdDRSW_ComputeOffsetsKernel_Smem(
        const int* __restrict__ node_ids,
        int*       __restrict__ node_offsets,
        int*       __restrict__ d_selected_examples,
        const int* __restrict__ node_row_off,
        int*       __restrict__ d_atomic_counters,
        int num_rows,
        int num_nodes)
{
    constexpr int NUM_WARPS = 8;  // BLOCK is always 256
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    __shared__ int s_base[SMEM_NODES];
    __shared__ int s_warp_prefix[SMEM_NODES * NUM_WARPS];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int nid = -1;
    if (row < num_rows) {
        nid = node_ids[row];
        if (nid < 0 || nid >= num_nodes) nid = -1;
    }

    // --- Step 1: per-warp per-node counts via warp matching ---
    for (int i = threadIdx.x; i < SMEM_NODES * NUM_WARPS; i += blockDim.x)
        s_warp_prefix[i] = 0;
    __syncthreads();

    unsigned active = __ballot_sync(0xFFFFFFFF, nid >= 0);
    unsigned match = 0;
    if (nid >= 0) {
        match = __match_any_sync(active, nid);
        int leader = __ffs(match) - 1;
        if (lane == leader)
            s_warp_prefix[nid * NUM_WARPS + warp_id] = __popc(match);
    }
    __syncthreads();

    // --- Step 2: prefix sum over warps + global reservation ---
    if (threadIdx.x < num_nodes) {
        int n = threadIdx.x;
        int total = 0;
        for (int w = 0; w < NUM_WARPS; w++) {
            int cnt = s_warp_prefix[n * NUM_WARPS + w];
            s_warp_prefix[n * NUM_WARPS + w] = total;
            total += cnt;
        }
        if (total > 0)
            s_base[n] = atomicAdd(&d_atomic_counters[n], total);
    }
    __syncthreads();

    // --- Step 3: deterministic rank = base + warp_prefix + intra_warp_rank ---
    if (nid < 0) return;

    int intra_warp_rank = __popc(match & ((1u << lane) - 1));
    int offset = s_base[nid]
               + s_warp_prefix[nid * NUM_WARPS + warp_id]
               + intra_warp_rank;

    node_offsets[row] = offset;
    d_selected_examples[node_row_off[nid] + offset] = row;
}

// Global-atomics fallback: num_nodes > SMEM_NODES
// Uses warp-level __match_any_sync to coalesce atomics — one atomicAdd per
// distinct node_id per warp instead of per thread.  Within each warp's group,
// lower lanes (lower row indices) get lower offsets, giving partial sort order.
__global__ void NodeIdDRSW_ComputeOffsetsKernel_Global(
        const int* __restrict__ node_ids,
        int*       __restrict__ node_offsets,
        int*       __restrict__ d_selected_examples,
        const int* __restrict__ node_row_off,
        int*       __restrict__ d_atomic_counters,
        int num_rows,
        int num_nodes)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int nid = -1;
    if (row < num_rows) {
        nid = node_ids[row];
        if (nid < 0 || nid >= num_nodes) nid = -1;
    }

    unsigned active = __ballot_sync(0xFFFFFFFF, nid >= 0);
    if (nid < 0) return;

    unsigned match_mask = __match_any_sync(active, nid);
    int lane = threadIdx.x & 31;
    int leader = __ffs(match_mask) - 1;
    int group_size = __popc(match_mask);

    int base_offset;
    if (lane == leader) {
        base_offset = atomicAdd(&d_atomic_counters[nid], group_size);
    }
    base_offset = __shfl_sync(match_mask, base_offset, leader);

    unsigned lower_mask = (1u << lane) - 1;
    int offset = base_offset + __popc(match_mask & lower_mask);

    node_offsets[row] = offset;
    d_selected_examples[node_row_off[nid] + offset] = row;
}

// ── Two-pass ComputeOffsets for fully sorted inter-block order ──
// Layout: d_block_counts[nid * num_blocks + block_id] (node-major)
//         so each node's block counts are contiguous for prefix sum.
//
// Pass 1: each block counts how many rows it has per node.
template <int SMEM_NODES>
__global__ void NodeIdDRSW_ComputeOffsets_CountKernel(
        const int* __restrict__ node_ids,
        int*       __restrict__ d_block_counts,
        int num_rows,
        int num_nodes,
        int num_blocks)
{
    __shared__ int s_counts[SMEM_NODES];

    for (int i = threadIdx.x; i < num_nodes; i += blockDim.x)
        s_counts[i] = 0;
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int nid = -1;
    if (row < num_rows) {
        nid = node_ids[row];
        if (nid < 0 || nid >= num_nodes) nid = -1;
    }

    if (nid >= 0)
        atomicAdd(&s_counts[nid], 1);
    __syncthreads();

    // Write in node-major layout: d_block_counts[nid * num_blocks + blockIdx.x]
    for (int i = threadIdx.x; i < num_nodes; i += blockDim.x) {
        d_block_counts[i * num_blocks + blockIdx.x] = s_counts[i];
    }
}

// Between passes: CUB ExclusiveSum on the full flat array.
// Since layout is [node0_blk0, node0_blk1, ..., node1_blk0, node1_blk1, ...],
// the flat prefix sum naturally gives per-node prefix sums across blocks,
// plus node-to-node offsets accumulate automatically.

// Pass 2: scatter with deterministic ranks using prefix-summed block offsets.
// WRITE_SELECTED=false computes node_offsets only and skips the
// d_selected_examples store (two-pass offsets-only variant).
// WRITE_LABELS=true additionally scatters the per-example label (and, under
// DEDUP_BOOTSTRAP, dup count) into SRDW layout at the same slot, so the SRDW
// histogram can read them coalesced without d_selected_examples
// (NODE_ID_DRSW_SRDW_LABEL_HIST). The label/dup pointers default to null and the
// branch is dead when WRITE_LABELS=false, leaving existing paths unchanged.
template <int SMEM_NODES, bool WRITE_SELECTED, bool WRITE_LABELS = false>
__global__ void NodeIdDRSW_ComputeOffsets_ScatterKernel(
        const int* __restrict__ node_ids,
        int*       __restrict__ node_offsets,
        int*       __restrict__ d_selected_examples,
        const int* __restrict__ node_row_off,
        const int* __restrict__ d_block_offsets,
        int num_rows,
        int num_nodes,
        int num_blocks,
        const int* __restrict__ d_labels = nullptr,
        const int* __restrict__ d_dup = nullptr,
        int*       __restrict__ d_labels_srdw = nullptr,
        int*       __restrict__ d_dup_srdw = nullptr)
{
    constexpr int NUM_WARPS = 8;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    __shared__ int s_base[SMEM_NODES];
    __shared__ int s_warp_prefix[SMEM_NODES * NUM_WARPS];

    // Load this block's base offsets: d_block_offsets[nid * num_blocks + blockIdx.x]
    // Then subtract node_row_off[nid] so offset is relative to node start
    for (int i = threadIdx.x; i < num_nodes; i += blockDim.x)
        s_base[i] = d_block_offsets[i * num_blocks + blockIdx.x]
                   - node_row_off[i];

    for (int i = threadIdx.x; i < SMEM_NODES * NUM_WARPS; i += blockDim.x)
        s_warp_prefix[i] = 0;
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int nid = -1;
    if (row < num_rows) {
        nid = node_ids[row];
        if (nid < 0 || nid >= num_nodes) nid = -1;
    }

    // Warp-level matching for deterministic intra-block ranking
    unsigned active = __ballot_sync(0xFFFFFFFF, nid >= 0);
    unsigned match = 0;
    if (nid >= 0) {
        match = __match_any_sync(active, nid);
        int leader = __ffs(match) - 1;
        if (lane == leader)
            s_warp_prefix[nid * NUM_WARPS + warp_id] = __popc(match);
    }
    __syncthreads();

    // Prefix sum over warps
    if (threadIdx.x < num_nodes) {
        int n = threadIdx.x;
        int total = 0;
        for (int w = 0; w < NUM_WARPS; w++) {
            int cnt = s_warp_prefix[n * NUM_WARPS + w];
            s_warp_prefix[n * NUM_WARPS + w] = total;
            total += cnt;
        }
    }
    __syncthreads();

    if (nid < 0) return;

    int intra_warp_rank = __popc(match & ((1u << lane) - 1));
    int offset = s_base[nid]
               + s_warp_prefix[nid * NUM_WARPS + warp_id]
               + intra_warp_rank;

    node_offsets[row] = offset;
    const int slot = node_row_off[nid] + offset;
    if (WRITE_SELECTED)
        d_selected_examples[slot] = row;
    if (WRITE_LABELS) {
        d_labels_srdw[slot] = d_labels[row];
#ifdef DEDUP_BOOTSTRAP
        d_dup_srdw[slot] = d_dup[row];
#endif
    }
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

    // ── Two-pass buffers (persistent across calls) ──
    static int* d_block_counts = nullptr;
    static int* d_block_offsets = nullptr;
    static int bc_capacity = 0;  // capacity in elements (blocks * nodes)

    int bc_needed = blocks * num_nodes;
    if (bc_needed > bc_capacity) {
        if (d_block_counts) { cudaFree(d_block_counts); cudaFree(d_block_offsets); }
        int alloc_elems = std::max(bc_needed, 65536);
        CUDA_CHECK(cudaMalloc(&d_block_counts,  alloc_elems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_block_offsets, alloc_elems * sizeof(int)));
        bc_capacity = alloc_elems;
    }

    // CUB temp storage for ExclusiveSum
    static void* d_cub_temp = nullptr;
    static size_t cub_temp_cap = 0;
    size_t cub_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, cub_temp_bytes, d_block_counts, d_block_offsets, bc_needed));
    if (cub_temp_bytes > cub_temp_cap) {
        if (d_cub_temp) cudaFree(d_cub_temp);
        cub_temp_cap = std::max(cub_temp_bytes, (size_t)(1 << 20));
        CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_cap));
    }

    // ── A/B/C benchmark: OLD vs warp-ranked vs two-pass (every depth, tree 0) ──
    // Opt-in only (COMPUTE_OFFSETS_AB=1): the extra kernel launches otherwise
    // inflate the timed ComputeOffsets / apply-projection region.
    static int bench_call = 0;
    static bool bench_header_printed = false;
    static const bool co_bench_enabled =
        [] { const char* e = getenv("COMPUTE_OFFSETS_AB"); return e && e[0] == '1'; }();
    if (co_bench_enabled && bench_call < 10 && num_nodes <= 256) {
        constexpr int WARMUP = 3;
        constexpr int ITERS  = 20;

        int* d_off_tmp = nullptr;
        int* d_sel_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_off_tmp, num_rows * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sel_tmp, total_rows * sizeof(int)));

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));

        // --- benchmark OLD (per-thread smem atomicAdd) ---
        for (int i = 0; i < WARMUP; i++) {
            CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
            NodeIdDRSW_ComputeOffsetsKernel_Smem_OldAtomicAdd<256><<<blocks, BLOCK>>>(
                d_node_ids, d_off_tmp, d_sel_tmp,
                d_node_row_off, d_atomic_counters, num_rows, num_nodes);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < ITERS; i++) {
            CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
            NodeIdDRSW_ComputeOffsetsKernel_Smem_OldAtomicAdd<256><<<blocks, BLOCK>>>(
                d_node_ids, d_off_tmp, d_sel_tmp,
                d_node_row_off, d_atomic_counters, num_rows, num_nodes);
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float old_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&old_ms, e0, e1));

        // --- benchmark WARP-RANKED (single pass, warp matching) ---
        for (int i = 0; i < WARMUP; i++) {
            CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
            NodeIdDRSW_ComputeOffsetsKernel_Smem<256><<<blocks, BLOCK>>>(
                d_node_ids, d_off_tmp, d_sel_tmp,
                d_node_row_off, d_atomic_counters, num_rows, num_nodes);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < ITERS; i++) {
            CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
            NodeIdDRSW_ComputeOffsetsKernel_Smem<256><<<blocks, BLOCK>>>(
                d_node_ids, d_off_tmp, d_sel_tmp,
                d_node_row_off, d_atomic_counters, num_rows, num_nodes);
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float warp_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&warp_ms, e0, e1));

        // --- benchmark TWO-PASS (count → prefix sum → scatter) ---
        for (int i = 0; i < WARMUP; i++) {
            NodeIdDRSW_ComputeOffsets_CountKernel<256><<<blocks, BLOCK>>>(
                d_node_ids, d_block_counts, num_rows, num_nodes, blocks);
            cub::DeviceScan::ExclusiveSum(
                d_cub_temp, cub_temp_bytes, d_block_counts, d_block_offsets, bc_needed);
            NodeIdDRSW_ComputeOffsets_ScatterKernel<256, true><<<blocks, BLOCK>>>(
                d_node_ids, d_off_tmp, d_sel_tmp,
                d_node_row_off, d_block_offsets, num_rows, num_nodes, blocks);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < ITERS; i++) {
            NodeIdDRSW_ComputeOffsets_CountKernel<256><<<blocks, BLOCK>>>(
                d_node_ids, d_block_counts, num_rows, num_nodes, blocks);
            cub::DeviceScan::ExclusiveSum(
                d_cub_temp, cub_temp_bytes, d_block_counts, d_block_offsets, bc_needed);
            NodeIdDRSW_ComputeOffsets_ScatterKernel<256, true><<<blocks, BLOCK>>>(
                d_node_ids, d_off_tmp, d_sel_tmp,
                d_node_row_off, d_block_offsets, num_rows, num_nodes, blocks);
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float twopass_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&twopass_ms, e0, e1));

        float old_avg = old_ms / ITERS;
        float warp_avg = warp_ms / ITERS;
        float twopass_avg = twopass_ms / ITERS;

        if (!bench_header_printed) {
            fprintf(stderr,
                "\n══ ComputeOffsets A/B/C: OLD(atomicAdd) vs WARP(1pass) vs 2PASS(sorted) ══\n"
                "  %-6s %-8s %-11s %-11s %-11s %-9s %-9s\n",
                "nodes", "rows", "OLD(ms)", "WARP(ms)", "2PASS(ms)", "W/O", "2P/O");
            bench_header_printed = true;
        }
        fprintf(stderr, "  %-6d %-8d %-11.4f %-11.4f %-11.4f %-9.2fx %-9.2fx\n",
            num_nodes, num_rows, old_avg, warp_avg, twopass_avg,
            old_avg / warp_avg, old_avg / twopass_avg);

        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaFree(d_off_tmp));
        CUDA_CHECK(cudaFree(d_sel_tmp));
        bench_call++;
    }

    // ── actual run ──
    // Two-pass (count -> prefix-sum -> scatter) vs OLD single-pass atomicAdd is
    // toggled by TWO_PASS (--config=two_pass, default on; --config=no_two_pass
    // disables). Both rebuild d_selected_examples (WRITE_SELECTED=true).
#if defined(TWO_PASS)
    if (num_nodes <= 256) {
        // Two-pass for fully sorted order
        NodeIdDRSW_ComputeOffsets_CountKernel<256><<<blocks, BLOCK>>>(
            d_node_ids, d_block_counts, num_rows, num_nodes, blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            d_cub_temp, cub_temp_bytes, d_block_counts, d_block_offsets, bc_needed));
        NodeIdDRSW_ComputeOffsets_ScatterKernel<256, true><<<blocks, BLOCK>>>(
            d_node_ids, d_node_offsets, d_selected_examples,
            d_node_row_off, d_block_offsets, num_rows, num_nodes, blocks);
    } else {
        // Fallback: global atomics path for large node counts
        CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
        NodeIdDRSW_ComputeOffsetsKernel_Global<<<blocks, BLOCK>>>(
            d_node_ids, d_node_offsets, d_selected_examples,
            d_node_row_off, d_atomic_counters,
            num_rows, num_nodes);
    }
#else
    if (num_nodes <= 256) {
        // Old single-pass atomicAdd
        CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
        NodeIdDRSW_ComputeOffsetsKernel_Smem_OldAtomicAdd<256><<<blocks, BLOCK>>>(
            d_node_ids, d_node_offsets, d_selected_examples,
            d_node_row_off, d_atomic_counters, num_rows, num_nodes);
    } else {
        // Fallback: global atomics path for large node counts
        CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));
        NodeIdDRSW_ComputeOffsetsKernel_Global<<<blocks, BLOCK>>>(
            d_node_ids, d_node_offsets, d_selected_examples,
            d_node_row_off, d_atomic_counters,
            num_rows, num_nodes);
    }
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify sorted order (first tree only)
    static bool verify_done = false;
    if (!verify_done && num_nodes > 1) {
        verify_done = true;
        bool ok = NodeIdDRSW_VerifySorted(d_selected_examples, d_node_row_off, num_nodes);
        if (!ok) {
            // Dump first few elements of each node for debugging
            std::vector<int> h_row_off(num_nodes + 1);
            CUDA_CHECK(cudaMemcpy(h_row_off.data(), d_node_row_off,
                       (num_nodes + 1) * sizeof(int), cudaMemcpyDeviceToHost));
            for (int n = 0; n < std::min(num_nodes, 4); n++) {
                int start = h_row_off[n];
                int cnt = h_row_off[n + 1] - start;
                int show = std::min(cnt, 20);
                std::vector<int> h_sel(show);
                CUDA_CHECK(cudaMemcpy(h_sel.data(), d_selected_examples + start,
                           show * sizeof(int), cudaMemcpyDeviceToHost));
                fprintf(stderr, "  Node %d [off=%d, cnt=%d]: ", n, start, cnt);
                for (int i = 0; i < show; i++) fprintf(stderr, "%d ", h_sel[i]);
                fprintf(stderr, "...\n");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2b. COMPACT ACTIVE ROWS — CUB select active row indices for histogram
// ═══════════════════════════════════════════════════════════════════════

// Build flags + row indices for CUB DeviceSelect::Flagged
__global__ void NodeIdDRSW_BuildFlagsKernel(
        const int* __restrict__ node_ids,
        int*       __restrict__ d_row_indices,
        uint8_t*   __restrict__ d_flags,
        int num_rows)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows) return;
    d_row_indices[r] = r;
    d_flags[r] = (node_ids[r] >= 0) ? 1 : 0;
}

inline int NodeIdDRSW_CompactActiveRows(
        const int* d_node_ids,
        int* d_active_rows,
        int num_rows)
{
    constexpr int BLOCK = 256;

    static int* d_row_indices = nullptr;
    static uint8_t* d_flags = nullptr;
    static int buf_cap = 0;
    if (num_rows > buf_cap) {
        if (d_row_indices) { cudaFree(d_row_indices); cudaFree(d_flags); }
        buf_cap = num_rows;
        CUDA_CHECK(cudaMalloc(&d_row_indices, num_rows * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_flags, num_rows * sizeof(uint8_t)));
    }

    int blocks = (num_rows + BLOCK - 1) / BLOCK;
    NodeIdDRSW_BuildFlagsKernel<<<blocks, BLOCK>>>(
        d_node_ids, d_row_indices, d_flags, num_rows);
    CUDA_CHECK(cudaGetLastError());

    static int* d_num_selected = nullptr;
    if (!d_num_selected) {
        CUDA_CHECK(cudaMalloc(&d_num_selected, sizeof(int)));
    }

    static void* d_temp = nullptr;
    static size_t temp_cap = 0;
    size_t temp_bytes = 0;

    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_bytes, d_row_indices, d_flags,
        d_active_rows, d_num_selected, num_rows));

    if (temp_bytes > temp_cap) {
        if (d_temp) cudaFree(d_temp);
        temp_cap = std::max(temp_bytes, (size_t)(1 << 20));
        CUDA_CHECK(cudaMalloc(&d_temp, temp_cap));
    }

    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp, temp_bytes, d_row_indices, d_flags,
        d_active_rows, d_num_selected, num_rows));
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_num_active;
    CUDA_CHECK(cudaMemcpy(&h_num_active, d_num_selected, sizeof(int),
                          cudaMemcpyDeviceToHost));
    return h_num_active;
}


// write_labels=true additionally scatters the per-example label (and, under
// DEDUP_BOOTSTRAP, dup count) into SRDW layout at node_row_off[nid] + local_r,
// for the NODE_ID_DRSW_SRDW_LABEL_HIST path on large node counts (> 256) where
// the two-pass scatter is not used. The extra pointers default to null and the
// branch is dead when write_labels=false, leaving existing behavior unchanged.
__global__ void NodeIdDRSW_ComputeOffsetsOnlyKernel(
        const int* __restrict__ node_ids,
        int*       __restrict__ node_offsets,
        int*       __restrict__ d_atomic_counters,
        int num_rows,
        int num_nodes,
        const int* __restrict__ node_row_off = nullptr,
        const int* __restrict__ d_labels = nullptr,
        const int* __restrict__ d_dup = nullptr,
        int*       __restrict__ d_labels_srdw = nullptr,
        int*       __restrict__ d_dup_srdw = nullptr,
        bool write_labels = false)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int nid = -1;
    if (row < num_rows) {
        nid = node_ids[row];
        if (nid < 0 || nid >= num_nodes) nid = -1;
    }

    unsigned active = __ballot_sync(0xFFFFFFFF, nid >= 0);
    if (nid < 0) return;

    unsigned match_mask = __match_any_sync(active, nid);
    int lane = threadIdx.x & 31;
    int leader = __ffs(match_mask) - 1;
    int group_size = __popc(match_mask);

    int base_offset;
    if (lane == leader) {
        base_offset = atomicAdd(&d_atomic_counters[nid], group_size);
    }
    base_offset = __shfl_sync(match_mask, base_offset, leader);

    unsigned lower_mask = (1u << lane) - 1;
    int local_r = base_offset + __popc(match_mask & lower_mask);

    node_offsets[row] = local_r;

    if (write_labels) {
        const int slot = node_row_off[nid] + local_r;
        d_labels_srdw[slot] = d_labels[row];
#ifdef DEDUP_BOOTSTRAP
        d_dup_srdw[slot] = d_dup[row];
#endif
    }
}

// The d_labels/d_dup/d_labels_srdw/d_dup_srdw pointers are used only in the
// NODE_ID_DRSW_SRDW_LABEL_HIST build: there the scatter also writes the per-example
// label (and dup count) into SRDW layout so the SRDW histogram can read them
// coalesced without d_selected_examples. They default to null and are ignored in
// every other build.
inline void NodeIdDRSW_ComputeOffsetsOnly(
        int* d_node_offsets,
        const int* d_node_ids,
        const int* d_node_row_off,
        int num_rows,
        int num_nodes,
        const int* d_labels = nullptr,
        const int* d_dup = nullptr,
        int* d_labels_srdw = nullptr,
        int* d_dup_srdw = nullptr)
{
    constexpr int BLOCK = 256;
    int blocks = (num_rows + BLOCK - 1) / BLOCK;

#ifdef NODE_ID_DRSW_SRDW_LABEL_HIST
    constexpr bool kWriteLabels = true;
#else
    constexpr bool kWriteLabels = false;
#endif

#if defined(TWO_PASS)
    // Two-pass offsets-only: same count -> prefix-sum -> scatter as the do-both
    // path, but the scatter skips the d_selected_examples store
    // (WRITE_SELECTED=false). Isolates the cost of that one store versus the
    // do-both two-pass. num_nodes > 256 falls through to the single warp pass.
    if (num_nodes <= 256) {
        static int* d_block_counts = nullptr;
        static int* d_block_offsets = nullptr;
        static int bc_capacity = 0;
        int bc_needed = blocks * num_nodes;
        if (bc_needed > bc_capacity) {
            if (d_block_counts) { cudaFree(d_block_counts); cudaFree(d_block_offsets); }
            int alloc_elems = std::max(bc_needed, 65536);
            CUDA_CHECK(cudaMalloc(&d_block_counts,  alloc_elems * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_block_offsets, alloc_elems * sizeof(int)));
            bc_capacity = alloc_elems;
        }
        static void* d_cub_temp = nullptr;
        static size_t cub_temp_cap = 0;
        size_t cub_temp_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, cub_temp_bytes, d_block_counts, d_block_offsets, bc_needed));
        if (cub_temp_bytes > cub_temp_cap) {
            if (d_cub_temp) cudaFree(d_cub_temp);
            cub_temp_cap = std::max(cub_temp_bytes, (size_t)(1 << 20));
            CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_cap));
        }
        NodeIdDRSW_ComputeOffsets_CountKernel<256><<<blocks, BLOCK>>>(
            d_node_ids, d_block_counts, num_rows, num_nodes, blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            d_cub_temp, cub_temp_bytes, d_block_counts, d_block_offsets, bc_needed));
        NodeIdDRSW_ComputeOffsets_ScatterKernel<256, false, kWriteLabels><<<blocks, BLOCK>>>(
            d_node_ids, d_node_offsets, /*d_selected_examples=*/nullptr,
            d_node_row_off, d_block_offsets, num_rows, num_nodes, blocks,
            d_labels, d_dup, d_labels_srdw, d_dup_srdw);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
#endif

    static int* d_atomic_counters = nullptr;
    static int counters_capacity = 0;
    if (num_nodes > counters_capacity) {
        if (d_atomic_counters) cudaFree(d_atomic_counters);
        int alloc_size = std::max(num_nodes, 65536);
        CUDA_CHECK(cudaMalloc(&d_atomic_counters, alloc_size * sizeof(int)));
        counters_capacity = alloc_size;
    }
    CUDA_CHECK(cudaMemset(d_atomic_counters, 0, num_nodes * sizeof(int)));

    NodeIdDRSW_ComputeOffsetsOnlyKernel<<<blocks, BLOCK>>>(
        d_node_ids, d_node_offsets, d_atomic_counters,
        num_rows, num_nodes,
        d_node_row_off, d_labels, d_dup, d_labels_srdw, d_dup_srdw, kWriteLabels);
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


// ═══════════════════════════════════════════════════════════════════════
// 4. NODE COUNTS — derive per-node row counts from d_node_ids
// ═══════════════════════════════════════════════════════════════════════

// Warp-level coalesced: one atomicAdd per distinct node_id per warp.
__global__ void NodeIdDRSW_CountRowsKernel(
        const int* __restrict__ node_ids,
        int*       __restrict__ counts,
        int num_rows,
        int num_nodes)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int nid = -1;
    if (row < num_rows) {
        nid = node_ids[row];
        if (nid < 0 || nid >= num_nodes) nid = -1;
    }

    unsigned active = __ballot_sync(0xFFFFFFFF, nid >= 0);
    if (nid < 0) return;

    unsigned match = __match_any_sync(active, nid);
    int lane = threadIdx.x & 31;
    int leader = __ffs(match) - 1;

    if (lane == leader)
        atomicAdd(&counts[nid], __popc(match));
}

inline std::vector<int> NodeIdDRSW_CountRows(
        const int* d_node_ids,
        int num_rows,
        int num_nodes)
{
    constexpr int BLOCK = 256;
    int* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counts, 0, num_nodes * sizeof(int)));

    int blocks = (num_rows + BLOCK - 1) / BLOCK;
    NodeIdDRSW_CountRowsKernel<<<blocks, BLOCK>>>(
        d_node_ids, d_counts, num_rows, num_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_counts(num_nodes);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
               num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_counts);
    return h_counts;
}


// ═══════════════════════════════════════════════════════════════════════
// 5. VERIFY SORTED — check that d_selected_examples is sorted per node
//    after ComputeOffsets.  Call from host for debugging.
// ═══════════════════════════════════════════════════════════════════════

__global__ void NodeIdDRSW_CheckSortedKernel(
        const int* __restrict__ d_selected_examples,
        const int* __restrict__ node_row_off,
        int*       __restrict__ d_violations,
        int num_nodes)
{
    int node = blockIdx.x;
    if (node >= num_nodes) return;

    int start = node_row_off[node];
    int count = node_row_off[node + 1] - start;

    for (int i = threadIdx.x + 1; i < count; i += blockDim.x) {
        if (d_selected_examples[start + i] < d_selected_examples[start + i - 1]) {
            atomicAdd(&d_violations[0], 1);
            atomicAdd(&d_violations[1 + node], 1);
        }
    }
}

inline bool NodeIdDRSW_VerifySorted(
        const int* d_selected_examples,
        const int* d_node_row_off,
        int num_nodes)
{
    int* d_violations;
    CUDA_CHECK(cudaMalloc(&d_violations, (1 + num_nodes) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_violations, 0, (1 + num_nodes) * sizeof(int)));

    constexpr int BLOCK = 256;
    NodeIdDRSW_CheckSortedKernel<<<num_nodes, BLOCK>>>(
        d_selected_examples, d_node_row_off, d_violations, num_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_violations(1 + num_nodes);
    CUDA_CHECK(cudaMemcpy(h_violations.data(), d_violations,
               (1 + num_nodes) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_violations);

    if (h_violations[0] > 0) {
        std::cerr << "[VerifySorted] FAILED: " << h_violations[0]
                  << " sort violations across " << num_nodes << " nodes\n";
        for (int n = 0; n < num_nodes; ++n) {
            if (h_violations[1 + n] > 0)
                std::cerr << "  Node " << n << ": "
                          << h_violations[1 + n] << " violations\n";
        }
        return false;
    }
    std::cerr << "[VerifySorted] PASSED: all " << num_nodes
              << " nodes have sorted elements\n";
    return true;
}
