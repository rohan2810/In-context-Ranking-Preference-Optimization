#!/bin/bash
# Usage: ./scripts/train.sh <dataset> <model> <loss> [additional args]
#

set -e

CUDA_DEVICE=${CUDA_DEVICE:-0}
WANDB_PROJECT=${WANDB_PROJECT:-"irpo"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
BATCH_SIZE=${BATCH_SIZE:-64}
GRAD_ACCUM=${GRAD_ACCUM:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
N_EPOCHS=${N_EPOCHS:-5}
BETA=${BETA:-1.0}

DATASET=${1:-"redial"}
MODEL=${2:-"llama3b"}
LOSS=${3:-"rdpo"}

# Dataset type: "reddit" for recommendation, "nlp" for QA
case $DATASET in
    redial|inspired) DS_TYPE="reddit" ;;
    *) DS_TYPE="nlp" ;;
esac

TRAIN_DATA="datasets/${DATASET}_train.csv"
EXP_NAME="${MODEL}_${DATASET}_${LOSS}"

echo "========================================"
echo "IRPO Training"
echo "========================================"
echo "Dataset: $DATASET (type: $DS_TYPE)"
echo "Model: $MODEL"
echo "Loss: $LOSS"
echo "Train data: $TRAIN_DATA"
echo "Experiment: $EXP_NAME"
echo "========================================"

# Build wandb args
WANDB_ARGS=""
if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS="wandb.enabled=true wandb.project=$WANDB_PROJECT wandb.entity=$WANDB_ENTITY"
else
    WANDB_ARGS="wandb.enabled=false"
fi

# Run training
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 -u train.py \
    $WANDB_ARGS \
    embed_dirs="$TRAIN_DATA" \
    datasets=[$DS_TYPE] \
    model=$MODEL \
    loss=$LOSS \
    loss.beta=$BETA \
    exp_name=$EXP_NAME \
    gradient_accumulation_steps=$GRAD_ACCUM \
    batch_size=$BATCH_SIZE \
    eval_batch_size=$EVAL_BATCH_SIZE \
    n_epochs=$N_EPOCHS \
    "$@"

