#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit

##### 4bit test #####
python3 -u test_gsm.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --chat_template "llama-3.1" \
  --max_seq_length 512 \
  --no_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --no_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/test-gsm8k-0shot-nocot-Llama-3.1-8B-Instruct-NON-4BIT.log

python3 -u test_mmlu.py \
  --model_name "meta-llama/meta-llama/Llama-3.1-8B-Instruct" \
  --chat_template "llama" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 900 \
  --task_names haerae \
  --no_cot \
  --n_shot 0 \
  --batch_size 32 \
  2>&1 | tee ../log/test-haerae-0shot-nocot-Llama-3.1-8B-Instruct-NON-4BIT.log
