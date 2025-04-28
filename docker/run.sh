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

python3 -u test_lm_eval.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct" \
  --chat_template "llama-3.2" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --single_gpu \
  --task_names agieval_math \
  --no_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/agieval/test-gsm8k-0shot-nocot-Llama-3.2-3B-Instruct.log