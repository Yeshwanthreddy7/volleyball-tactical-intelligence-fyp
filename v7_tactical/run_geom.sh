#!/bin/bash
# Fires after csv_geom extraction completes: rebuild cache -> retrain binary
# (geometric features) -> (vp7 detection launched separately).
set -e
cd "$(dirname "$0")/.."

echo "=== [1/2] rebuild geometric cache ==="
python -u v7_tactical/prep_cache.py --csv-dir v7_tactical/csv_geom \
  --out v7_tactical/cache/binary_cache.npz

echo "=== [2/2] train binary (geometric) ==="
python -u v7_tactical/train_binary.py --epochs 100 --lovo-ensemble 3 \
  --ensemble 5 --patience 25 --tag binary_geom

echo "=== geom training done ==="
ls -la v7_tactical/models/binary_geom*.pt
