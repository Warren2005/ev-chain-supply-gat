#!/bin/bash
# Quick training script with default parameters

echo "Starting ST-GAT production training..."
echo ""

python scripts/train_st_gat.py \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --weight_decay 0.00001 \
    --patience 20 \
    --device cpu

echo ""
echo "Training complete!"