#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
# python3 -u test_gsm.py \
#   --model_name "meta-llama/LLama-3.2-3B-Instruct" \
#   --chat_template "llama-3.1" \
#   --load_in_4bit \
#   --max_seq_length 512 \
#   --random_seed 42 \
#   --dataset openai/gsm8k \
#   --subset main \
#   --split test \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 16 \
#   2>&1 | tee ../log/rerun-gsm8k-0shot/gsm8k-test-0shot-nocot-llama-3.2-3b-instruct.log

# python3 -u test_gsm.py \
#   --model_name "meta-llama/LLama-3.1-8B-Instruct" \
#   --chat_template "llama-3.1" \
#   --load_in_4bit \
#   --max_seq_length 512 \
#   --random_seed 42 \
#   --dataset openai/gsm8k \
#   --subset main \
#   --split test \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 12 \
#   2>&1 | tee ../log/rerun-gsm8k-0shot/gsm8k-test-0shot-nocot-llama-3.1-8b-instruct.log

# python3 -u test_gsm.py \
#   --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
#   --chat_template "mistral" \
#   --load_in_4bit \
#   --max_seq_length 512 \
#   --random_seed 42 \
#   --dataset openai/gsm8k \
#   --subset main \
#   --split test \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 12 \
#   2>&1 | tee ../log/rerun-gsm8k-0shot/gsm8k-test-0shot-nocot-mistral-7b-instruct.log

# python3 -u test_gsm.py \
#   --model_name "mistralai/Mistral-Nemo-Instruct-2407" \
#   --chat_template "mistral" \
#   --load_in_4bit \
#   --max_seq_length 512 \
#   --random_seed 42 \
#   --dataset openai/gsm8k \
#   --subset main \
#   --split test \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 8 \
#   2>&1 | tee ../log/rerun-gsm8k-0shot/gsm8k-test-0shot-nocot-mistral-nemo-instruct.log

# python3 -u test_gsm.py \
#   --model_name "microsoft/Phi-4-mini-instruct" \
#   --chat_template "phi" \
#   --load_in_4bit \
#   --max_seq_length 512 \
#   --random_seed 42 \
#   --dataset openai/gsm8k \
#   --subset main \
#   --split test \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 16 \
#   2>&1 | tee ../log/rerun-gsm8k-0shot/gsm8k-test-0shot-nocot-phi-4-mini-instruct.log

# python3 -u test_gsm.py \
#   --model_name "microsoft/phi-4" \
#   --chat_template "phi" \
#   --load_in_4bit \
#   --max_seq_length 512 \
#   --random_seed 42 \
#   --dataset openai/gsm8k \
#   --subset main \
#   --split test \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 8 \
#   2>&1 | tee ../log/rerun-gsm8k-0shot/gsm8k-test-0shot-nocot-phi-4.log
