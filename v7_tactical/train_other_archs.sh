#!/bin/bash
cd "$(dirname "$0")/.."
for a in tcn bilstm mamba; do
  echo "=========== $a ==========="
  nl=4; [ "$a" = bilstm ] && nl=2
  python -u v7_tactical/train_binary.py --arch $a --tag binary_$a \
    --epochs 100 --lovo-ensemble 2 --ensemble 3 --patience 20 --n_layers $nl \
    2>&1 | grep -E "POOLED|balanced_accuracy|macro_f1|roc_auc|attack_f1|defense_f1|servable|seed[0-9]:"
done
echo "ALL OTHER ARCHS DONE"
