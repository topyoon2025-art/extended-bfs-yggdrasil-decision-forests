Benchmarks run 2026-10-26 on an NVIDIA L40S. Depth 7, 5 trees x 2 alternating repeats per build (10 trees),
seed 1, sampling without replacement at bootstrap_size_ratio=1.0, density factor 1.5 (SRDW/DRSW sweeps).
Datasets: HIGGS1 (11,000,000 x 28, tfrecord) and 22000000x28.csv.

srdw_vs_drsw_summary.csv            SRDW before (bfs_only+use_gpu, partition-based example mgmt) vs
                                    SRDW after (+node_id_drsw+offsets_only+scattered_label_hist+nid_srdw_proj) vs
                                    DRSW (+node_id_drsw+offsets_only+scattered_label_hist). Random projections per node.
srdw_vs_drsw_symmetric_summary.csv  same three builds with --config=symmetric_proj.
cyclic_pairs_summary.csv            node-id DRSW path + pair_proj_cyclic (pairs AB, BC, ..., last-first) with
                                    pair_proj (ungrouped), pair_proj_grouped (grouped), + pair_proj_fused (fused).

Columns (ms per tree, median over the 10 trees). First column = dataset.
  AP + Example Mgmt + Eval Proj + Node Train + Overhead = tree total (exactly, per row)
  AP           = apply-projection kernel
  Example Mgmt = compute offsets + partition / split update
  Eval Proj    = split finding, INCLUDING the histogram build
  Overhead     = everything else: GPU setup + GPU cleanup + projection flatten + host-to-device + gaps
  Extra Selected Examples Write = SRDW-after compute offsets minus DRSW compute offsets
  AP vs Ungrouped (cyclic only) = AP / AP of the ungrouped build on the same dataset

axis_pairs_ungrouped_summary.csv      28 axis-aligned + 28 cyclic pairs (pair_proj + pair_proj_cyclic), ungrouped,
                                      for SRDW before / SRDW after / DRSW.
axis_pairs_grouped_fused_summary.csv  same projections, pair_proj_grouped + pair_proj_fused (SRDW builds use the
                                      SRDW fused kernel, DRSW the DRSW fused kernel). AP vs Ungrouped = AP / AP of the
                                      same build in axis_pairs_ungrouped_summary.csv. Verified bit-identical projected
                                      values (PAIR_FUSED_VERIFY=1) and byte-identical trees vs unfused grouped builds.
