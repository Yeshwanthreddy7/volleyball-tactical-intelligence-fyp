#!/bin/bash
# Train the 4 deep-learning Attack-vs-Defense models (LOVO evaluation + final fit).
cd "$(dirname "$0")/../.."
P="python -u BINARY_ATTACK_DEFENSE/code/train_binary_dl.py --epochs 100 --lovo-ensemble 3 --ensemble 3 --patience 22"
$P --arch inceptiontime --tag inceptiontime --n_layers 6
$P --arch tcn           --tag tcn           --n_layers 4
$P --arch bilstm        --tag bilstm        --n_layers 2
$P --arch mamba         --tag mamba         --n_layers 4
echo "ALL DL DONE"
