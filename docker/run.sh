#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
# python3 -u test_mmlu.py \
#   --model_name "google/gemma-2-27b-it" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae \
#   --no_cot \
#   --n_shot 8 \
#   --batch_size 2 \
#   2>&1 | tee ../log/rerun-for-compute-time/haerae-test-8shot-nocot-gemma-2-27b-it

python3 -u test_gsm.py \
  --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
  --chat_template "mistral" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --use_cot \
  --n_shot 0 \
  --batch_size 1 \
  2>&1 | tee ../log/rerun-for-compute-time/gsm8k-test-0shot-cot-Mistral-7B-Instruct-v0.3