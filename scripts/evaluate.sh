#!/bin/bash
# IRPO Evaluation Script
# Usage: ./scripts/evaluate.sh <dataset> <model> [checkpoint_path] [additional args]
#

set -e

CUDA_DEVICE=${CUDA_DEVICE:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}

DATASET=${1:-"redial"}
MODEL=${2:-"llama3b"}
CHECKPOINT=${3:-""}
shift 3 2>/dev/null || true

# Dataset type: "reddit" for recommendation, "nlp" for QA
case $DATASET in
    redial|inspired) DS_TYPE="reddit" ;;
    *) DS_TYPE="nlp" ;;
esac

EVAL_DATA="datasets/${DATASET}_eval.csv"

echo "========================================"
echo "IRPO Evaluation"
echo "========================================"
echo "Dataset: $DATASET (type: $DS_TYPE)"
echo "Model: $MODEL"
echo "Eval data: $EVAL_DATA"
echo "Checkpoint: ${CHECKPOINT:-'(base model)'}"
echo "========================================"

# Build checkpoint arg
ARCHIVE_ARG=""
if [ -n "$CHECKPOINT" ]; then
    ARCHIVE_ARG="model.archive=$CHECKPOINT"
fi

# Run evaluation
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 -u evaluate.py \
    embed_dirs="$EVAL_DATA" \
    datasets=[$DS_TYPE] \
    model=$MODEL \
    eval_batch_size=$EVAL_BATCH_SIZE \
    n_epochs=1 \
    $ARCHIVE_ARG \
    "$@"

