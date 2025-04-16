#!/bin/bash

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Default parameters
BATCH_SIZE=64
EPOCHS=50
LEARNING_RATE=0.001
WEIGHT_DECAY=0.01

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --batch_size)
        BATCH_SIZE="$2"
        shift
        shift
        ;;
        --epochs)
        EPOCHS="$2"
        shift
        shift
        ;;
        --lr)
        LEARNING_RATE="$2"
        shift
        shift
        ;;
        --weight_decay)
        WEIGHT_DECAY="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $key"
        exit 1
        ;;
    esac
done

# Print training configuration
echo "Starting distributed training with $NUM_GPUS GPUs"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Weight decay: $WEIGHT_DECAY"

# Run the distributed training script
python ddp_train.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY
