#!/bin/bash

set -ex

model_dir=/network/scratch/c/caleb.moses/group-project/models
model=llama-2-70b-chat  # Set your model here

if [ "$model" == "llama-2-7b-chat" ]; then
    num_gpus=1
elif [ "$model" == "llama-2-13b-chat" ]; then
    num_gpus=2
elif [ "$model" == "llama-2-70b-chat" ]; then
    num_gpus=8
else
    echo "Unknown model: $model"
    exit 1
fi

torchrun --nproc_per_node $num_gpus \
    llama/example_text_completion.py \
    --ckpt_dir $model_dir/$model \
    --tokenizer_path $model_dir/tokenizer.model \
    --max_seq_len 128 \
    --max_batch_size 4
