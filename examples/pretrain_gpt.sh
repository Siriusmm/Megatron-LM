#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/root/epfs/Megatron-LM
CHECKPOINT_PATH=/root/epfs/model_one
VOCAB_FILE=/root/epfs/Megatron-LM/gpt2-vocab.json
MERGE_FILE=/root/epfs/Megatron-LM/gpt2-merges.txt
DATA_PATH=/root/epfs/dataset/enwiki-latest-pages-articles

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 20 \
    --global-batch-size 320 \
    --lr 0.00015 \
    --train-iters 50000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun --nproc_per_node 8 /root/epfs/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
