#!/usr/bin/env bash
# DHR-ARIMA for Table 4 (IJF R2): choose (K1, K2) by summed AICc on 20 random series
# (first test window), then run all windows with the chosen pair.
# Usage: bash baselines/run_r2_dhr.sh   (from the repo root; ~2.5-3.5 h on an M1 Pro)
set -euo pipefail
PY=baselines/.venv/bin/python
SCRIPT=baselines/r2_statistical_baselines.py
for ds in solar_nips electricity_nips traffic_nips; do
  $PY $SCRIPT select-k --dataset "$ds" --k-grid 2 4 6 --n-series 20 --history 1344
  read -r K1 K2 < <($PY -c "import json; b=json.load(open('results/r2/selectk_$ds.json'))['best']; print(b['k1'], b['k2'])")
  echo "[$ds] K1=$K1 K2=$K2"
  $PY $SCRIPT run --model dhr --dataset "$ds" --k1 "$K1" --k2 "$K2" --history 1344
done
