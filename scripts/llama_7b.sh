#!/bin/bash

# Set common variables
model="pinkmanlove/llama-7b-hf"
sparsity_ratio=0.7
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python  main.py \
    --model $model \
    --cache_dir llm_weights/ \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --save $3 \
    --ww_metric alpha_peak \
    --ww_metric_cache ./data/llama-7b-hf/ \
    --epsilon $2
}


# llama-7b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude_ww" 0.3 "results/llama_7b/"
echo "Finished magnitude pruning method"

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda_ww" 0.2 "results/llama_7b/"
echo "Finished wanda pruning method"

# llama-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt_ww" 0.2 "results/llama_7b/"
echo "Finished sparsegpt pruning method"