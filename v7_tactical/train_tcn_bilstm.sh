#!/bin/bash
cd "$(dirname "$0")/.."
python -u v7_tactical/train_binary.py --arch tcn --tag binary_tcn --epochs 100 --lovo-ensemble 3 --ensemble 4 --patience 22 --n_layers 4 2>&1 | grep -E "===|balanced_acc|macro_f1|roc_auc|attack_f1|defense_f1|seed[0-9]:|servable|POOLED"
echo "=== BILSTM ==="
python -u v7_tactical/train_binary.py --arch bilstm --tag binary_bilstm --epochs 100 --lovo-ensemble 3 --ensemble 4 --patience 22 --n_layers 2 2>&1 | grep -E "balanced_acc|macro_f1|roc_auc|attack_f1|defense_f1|seed[0-9]:|servable"
echo "ARCHS DONE"
