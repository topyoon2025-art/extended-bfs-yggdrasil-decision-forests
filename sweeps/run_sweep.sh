#!/usr/bin/env bash
# Sampling-mode sweep for the oblique_apply_projection benchmark.
#
# Covers every sampling option now that the GPU sampler honors the flags:
#   repl    : bootstrap WITH replacement   (--bootstrap_training_dataset=true  --sampling_with_replacement=true  ratio=1)
#             -> draws N, dedups to ~0.632N unique, weighted by dup counts
#   noboot  : no bootstrap, all rows        (--bootstrap_training_dataset=false)                 -> N rows
#   wor1    : WITHOUT replacement, ratio=1  (--sampling_with_replacement=false ratio=1)          -> N rows (== noboot)
#   wor063  : WITHOUT replacement, ratio=.632(--sampling_with_replacement=false ratio=0.632)     -> ~0.632N rows, NO weights
#
# across three binaries:
#   nid           = node_id_drsw + offsets_only + scattered_label_hist (dedup ON, coalesced hist)
#   srdw_dedup    = SRDW baseline, dedup ON
#   srdw_nodedup  = SRDW baseline, dedup OFF
#
# Scenario label = <group><mode>, group A=nid, B=srdw_dedup, C=srdw_nodedup.
#
# Select with SWEEP (group letters, e.g. SWEEP=A, SWEEP=AB, SWEEP=ABC) or
# SCENARIOS="A_repl A_wor063" for an explicit subset. Config via env:
#   MODE=higgs|smoke  NUM_TREES  TREE_DEPTH  SEED  SMOKE_ROWS  SMOKE_COLS
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${MODE:-higgs}"
SWEEP="${SWEEP:-A}"
SCENARIOS="${SCENARIOS:-}"      # explicit label list overrides SWEEP
NUM_TREES="${NUM_TREES:-3}"
TREE_DEPTH="${TREE_DEPTH:-10}"
SEED="${SEED:-1}"
SMOKE_ROWS="${SMOKE_ROWS:-3000000}"
SMOKE_COLS="${SMOKE_COLS:-28}"
WOR_RATIO="${WOR_RATIO:-0.632}"  # subset fraction for the wor063 mode

BIN_DIR="$REPO/sweeps/bin"; OUT_DIR="$REPO/sweeps/results"; LOG_DIR="$REPO/sweeps/logs"
mkdir -p "$BIN_DIR" "$OUT_DIR" "$LOG_DIR"

if [[ "$MODE" == "smoke" ]]; then
  DATA_FLAGS=(--input_mode=trunk --rows="$SMOKE_ROWS" --cols="$SMOKE_COLS")
else
  DATA_FLAGS=(--input_mode=tfrecord --ds_path=/home/ubuntu/projects/dataset/HIGGS1 --label_col=target)
fi
COMMON_RUN=(--num_trees="$NUM_TREES" --tree_depth="$TREE_DEPTH" --seed="$SEED")

TARGET="//examples:oblique_apply_projection_main"
BUILT_BIN="bazel-bin/examples/oblique_apply_projection_main"

# label | binary | bootstrap | sampling_with_replacement | ratio
SCEN_ALL=(
  "A_repl|nid|true|true|1.0"
  "A_noboot|nid|false|true|1.0"
  "A_wor1|nid|true|false|1.0"
  "A_wor063|nid|true|false|$WOR_RATIO"
  "B_repl|srdw_dedup|true|true|1.0"
  "B_noboot|srdw_dedup|false|true|1.0"
  "B_wor1|srdw_dedup|true|false|1.0"
  "B_wor063|srdw_dedup|true|false|$WOR_RATIO"
  "C_repl|srdw_nodedup|true|true|1.0"
  "C_noboot|srdw_nodedup|false|true|1.0"
  "C_wor1|srdw_nodedup|true|false|1.0"
  "C_wor063|srdw_nodedup|true|false|$WOR_RATIO"
)

build_flags_for() {
  case "$1" in
    nid)          echo "--config=bfs_only --config=use_gpu --config=node_id_drsw --config=offsets_only --config=scattered_label_hist --define=split_smem=true" ;;
    srdw_dedup)   echo "--config=bfs_only --config=use_gpu --define=split_smem=true" ;;
    srdw_nodedup) echo "--config=bfs_only --config=use_gpu --config=no_dedup --define=split_smem=true" ;;
    *) echo ""; return 1 ;;
  esac
}

# Decide which scenarios to run.
selected=()
if [[ -n "$SCENARIOS" ]]; then
  for want in $SCENARIOS; do
    for s in "${SCEN_ALL[@]}"; do [[ "${s%%|*}" == "$want" ]] && selected+=("$s"); done
  done
else
  for s in "${SCEN_ALL[@]}"; do
    g="${s:0:1}"
    [[ "$SWEEP" == *"$g"* ]] && selected+=("$s")
  done
fi
if [[ ${#selected[@]} -eq 0 ]]; then
  echo "!!! no scenarios selected (SWEEP='$SWEEP' SCENARIOS='$SCENARIOS')"; exit 1
fi

build_bin() {
  local name="$1"; shift
  echo ">>> building [$name]: $*" | tee "$LOG_DIR/build_$name.log"
  if bazel build "$TARGET" "$@" >>"$LOG_DIR/build_$name.log" 2>&1; then
    cp -f "$BUILT_BIN" "$BIN_DIR/$name"; echo ">>> built  [$name]"
  else
    echo "!!! BUILD FAILED [$name] — see $LOG_DIR/build_$name.log"; return 1
  fi
}

# run_scenario <label> <binary> <bootstrap> <swr> <ratio>
run_scenario() {
  local label="$1" bin="$2" boot="$3" swr="$4" ratio="$5"
  echo "=== run [$label]  bin=$bin boot=$boot swr=$swr ratio=$ratio ==="
  rm -f per_tree_timing.csv per_depth_timing.csv
  "$BIN_DIR/$bin" "${DATA_FLAGS[@]}" "${COMMON_RUN[@]}" \
      --bootstrap_training_dataset="$boot" \
      --sampling_with_replacement="$swr" \
      --bootstrap_size_ratio="$ratio" \
      >"$LOG_DIR/run_$label.log" 2>&1
  local rc=$?
  [[ $rc -ne 0 ]] && echo "!!! RUN FAILED [$label] rc=$rc — see $LOG_DIR/run_$label.log"
  [[ -f per_tree_timing.csv ]] && cp -f per_tree_timing.csv "$OUT_DIR/$label.pertree.csv"
  grep "DEDUP_BOOTSTRAP:" "$LOG_DIR/run_$label.log" > "$OUT_DIR/$label.dedup.txt" 2>/dev/null || true
  { echo "label=$label"; echo "binary=$bin"; echo "bootstrap=$boot";
    echo "swr=$swr"; echo "ratio=$ratio"; echo "rc=$rc"; } > "$OUT_DIR/$label.meta.txt"
  echo "--- [$label] done (rc=$rc)"
}

echo "###############################################"
echo "# MODE=$MODE trees=$NUM_TREES depth=$TREE_DEPTH seed=$SEED  wor_ratio=$WOR_RATIO"
[[ "$MODE" == "smoke" ]] && echo "# synthetic rows=$SMOKE_ROWS cols=$SMOKE_COLS"
echo "# scenarios: $(for s in "${selected[@]}"; do printf '%s ' "${s%%|*}"; done)"
echo "###############################################"

# Build the unique binaries referenced by the selected scenarios.
declare -A seen_bin=()
for s in "${selected[@]}"; do
  IFS='|' read -r label bin boot swr ratio <<<"$s"
  [[ -n "${seen_bin[$bin]:-}" ]] && continue
  seen_bin[$bin]=1
  build_bin "$bin" $(build_flags_for "$bin") || exit 1
done

# Run them.
for s in "${selected[@]}"; do
  IFS='|' read -r label bin boot swr ratio <<<"$s"
  run_scenario "$label" "$bin" "$boot" "$swr" "$ratio"
done

echo; echo ">>> aggregating..."
python3 -I sweeps/aggregate.py "$OUT_DIR"
