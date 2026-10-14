#!/bin/bash
set -e
cd /home/ubuntu/projects/extended-bfs-yggdrasil-decision-forests
BIN=bazel-bin/examples/oblique_apply_projection_main
COMMON="--config=bfs_only --config=use_gpu --config=verbose_depth --config=node_id_drsw --define=split_smem=false"

echo "=== building srdw_label_hist ==="
bazel build //examples:oblique_apply_projection_main $COMMON --config=offsets_only --config=srdw_label_hist 2>&1 | tail -3
cp -f "$BIN" /tmp/ap_srdw_label

echo "=== building offsets_only (node_id+offset hist) ==="
bazel build //examples:oblique_apply_projection_main $COMMON --config=offsets_only 2>&1 | tail -3
cp -f "$BIN" /tmp/ap_offsets_only

echo "=== building nid_srdw_hist (SRDW baseline) ==="
bazel build //examples:oblique_apply_projection_main $COMMON --config=nid_srdw_hist 2>&1 | tail -3
cp -f "$BIN" /tmp/ap_nid_srdw

echo "=== ALL BUILDS DONE ==="
ls -la /tmp/ap_srdw_label /tmp/ap_offsets_only /tmp/ap_nid_srdw
