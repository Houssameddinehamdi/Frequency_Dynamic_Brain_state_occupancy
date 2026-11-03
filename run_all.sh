#!/usr/bin/env bash
# Statistics and figures from the exported session-level results.
#   bash run_all.sh <data folder> [output folder]
# <data folder> must contain pathA_results/ and clinical_processed/ (see docs/DATA_FORMAT.md).
set -euo pipefail
DATA_IN="${1:?usage: bash run_all.sh <data folder> [output folder]}"
OUT_IN="${2:-outputs}"
DATA="$(cd "$DATA_IN" && pwd)"
mkdir -p "$OUT_IN"; OUT="$(cd "$OUT_IN" && pwd)"
PY="${PYTHON:-python3}"
cd "$(dirname "$0")"
echo "== 1/3 primary mixed models"; "$PY" scripts/primary_models.py    --base "$DATA" --out "$OUT"
echo "== 2/3 clinical analyses";     "$PY" scripts/clinical_analysis.py --base "$DATA" --out "$OUT"
echo "== 3/3 figure panels";         "$PY" scripts/make_figures.py      --base "$DATA" --out "$OUT/figures" --clin "$OUT/clinical_analysis"
echo "Done. Tables: $OUT/tables | clinical: $OUT/clinical_analysis | figures: $OUT/figures"
