#!/usr/bin/env bash
# Runs the whole statistics + figure pipeline on a synthetic data set (no real data needed).
#   bash run_demo.sh
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python3}" scripts/make_demo_data.py --out demo_data
bash run_all.sh demo_data demo_outputs
