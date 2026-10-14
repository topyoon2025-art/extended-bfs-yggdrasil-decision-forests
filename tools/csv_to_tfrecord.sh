#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --csv=<path> --out=<path> [--label=<col>] [--build]

Convert a CSV file to TFRecord + binary dataspec for use with
  --input_mode=tfrecord --ds_path=<out>

Required:
  --csv=PATH     Path to input CSV file
  --out=PATH     Output path prefix (produces <out>.tfrecord and <out>.data_spec.pb)

Optional:
  --label=COL    Label column name (default: "target")
  --build        Build the CLI tools before running (otherwise assumes they exist)

Example:
  $0 --csv=data/train.csv --out=data/train --label="Cancer Status"
EOF
  exit 1
}

CSV=""
OUT=""
LABEL="target"
BUILD=false

for arg in "$@"; do
  case "$arg" in
    --csv=*)   CSV="${arg#*=}" ;;
    --out=*)   OUT="${arg#*=}" ;;
    --label=*) LABEL="${arg#*=}" ;;
    --build)   BUILD=true ;;
    --help|-h) usage ;;
    *)         echo "Unknown argument: $arg"; usage ;;
  esac
done

if [[ -z "$CSV" || -z "$OUT" ]]; then
  echo "Error: --csv and --out are required."
  usage
fi

CSV="$(realpath "$CSV")"
OUT_DIR="$(dirname "$(realpath -m "$OUT")")"
OUT_BASE="$(basename "$OUT")"
OUT_PREFIX="${OUT_DIR}/${OUT_BASE}"

if [[ ! -f "$CSV" ]]; then
  echo "Error: CSV file not found: $CSV"
  exit 1
fi
mkdir -p "$OUT_DIR"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if "$BUILD"; then
  echo "Building CLI tools..."
  (cd "$REPO_ROOT" && bazel build -c opt \
    //yggdrasil_decision_forests/cli:infer_dataspec \
    //yggdrasil_decision_forests/cli:convert_dataset)
fi

INFER="${REPO_ROOT}/bazel-bin/yggdrasil_decision_forests/cli/infer_dataspec"
CONVERT="${REPO_ROOT}/bazel-bin/yggdrasil_decision_forests/cli/convert_dataset"

for tool in "$INFER" "$CONVERT"; do
  if [[ ! -x "$tool" ]]; then
    echo "Error: $tool not found. Run with --build or build manually:"
    echo "  bazel build -c opt //yggdrasil_decision_forests/cli:infer_dataspec //yggdrasil_decision_forests/cli:convert_dataset"
    exit 1
  fi
done

GUIDE_FILE="$(mktemp)"
trap 'rm -f "$GUIDE_FILE"' EXIT
cat > "$GUIDE_FILE" <<EOF
column_guides { column_name_pattern: "${LABEL}" type: CATEGORICAL }
EOF

echo "Step 1/2: Inferring dataspec..."
"$INFER" \
  --dataset="csv:${CSV}" \
  --output="${OUT_PREFIX}.data_spec.pb" \
  --guide="$GUIDE_FILE" \
  --alsologtostderr

echo "Step 2/2: Converting CSV to TFRecord..."
"$CONVERT" \
  --input="csv:${CSV}" \
  --dataspec="${OUT_PREFIX}.data_spec.pb" \
  --output="tfrecord:${OUT_PREFIX}" \
  --alsologtostderr

echo ""
echo "Done. Output files:"
echo "  ${OUT_PREFIX}.data_spec.pb"
ls -lh "${OUT_PREFIX}"* 2>/dev/null | grep -v data_spec || true
echo ""
echo "Run with:"
echo "  --input_mode=tfrecord --ds_path=${OUT_PREFIX}"
