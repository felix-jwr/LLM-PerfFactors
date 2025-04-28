#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit

#### Examples ####

## GSM8k
# python3 -u test_gsm.py \
#   --model_name "meta-llama/Llama-3.1-8B-Instruct" \
#   --chat_template "llama-3.1" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --random_seed 42 \
#   --dataset openai/gsm8k \
#   --subset main \
#   --split test \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 16 \
#   2>&1 | tee ../log/examples/test-gsm8k-0shot-nocot-Llama-3.1-8B-Instruct.log

## HAERAE
# python3 -u test_lm_eval.py \
#   --model_name "meta-llama/Llama-3.1-8B-Instruct" \
#   --chat_template "llama-3.1" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --random_seed 42 \
#   --single_gpu \
#   --task_names haerae \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 16 \
#   2>&1 | tee ../log/examples/test-gsm8k-0shot-nocot-Llama-3.1-8B-Instruct.log

python3 -u test_gsm.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct" \
  --chat_template "llama-3.2" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 64 \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Llama-3.2-3B-Instruct.log

python3 -u test_gsm.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --chat_template "llama-3.1" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 32 \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Llama-3.1-8B-Instruct.log

python3 -u test_gsm.py \
  --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
  --chat_template "mistral" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 32 \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Mistral-7B-Instruct.log

python3 -u test_gsm.py \
  --model_name "mistralai/Mistral-Nemo-Instruct-2407" \
  --chat_template "mistral" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Mistral-Nemo-Instruct.log

python3 -u test_gsm.py \
  --model_name "microsoft/Phi-4-mini-instruct" \
  --chat_template "phi-4" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 48 \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Phi-4-mini-instruct.log

python3 -u test_gsm.py \
  --model_name "microsoft/phi-4" \
  --chat_template "phi-4" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 8  \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Phi-4.log

python3 -u test_gsm.py \
  --model_name "Qwen/Qwen2.5-1.5B" \
  --chat_template "qwen2.5" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 64 \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Qwen2.5-1.5B.log

python3 -u test_gsm.py \
  --model_name "Qwen/Qwen2.5-7B" \
  --chat_template "qwen2.5" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 32 \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Qwen2.5-7B.log

python3 -u test_gsm.py \
  --model_name "Qwen/Qwen2.5-14B" \
  --chat_template "qwen2.5" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset apple/GSM-Symbolic \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 12 \
  2>&1 | tee ../log/gsm_symbolic/test-gsm-symbolic-0shot-nocot-Qwen2.5-14B.log