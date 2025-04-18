#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test_gsm.py \
  --model_name "Qwen/Qwen2.5-1.5B" \
  --chat_template "qwen" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 64 \
  2>&1 | tee ../log/qwen-base/gsm8k-test-0shot-nocot-qwen-1.5b.log

python3 -u test_gsm.py \
  --model_name "Qwen/Qwen2.5-1.5B" \
  --chat_template "qwen" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --use_cot \
  --n_shot 0 \
  --batch_size 64 \
  2>&1 | tee ../log/qwen-base/gsm8k-test-0shot-cot-qwen-1.5b.log

python3 -u test_gsm.py \
  --model_name "Qwen/Qwen2.5-1.5B" \
  --chat_template "qwen" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 8 \
  --batch_size 48 \
  2>&1 | tee ../log/qwen-base/gsm8k-test-8shot-nocot-qwen-1.5b.log

python3 -u test_gsm.py \
  --model_name "Qwen/Qwen2.5-1.5B" \
  --chat_template "qwen" \
  --load_in_4bit \
  --max_seq_length 512 \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --use_cot \
  --n_shot 8 \
  --batch_size 48 \
  2>&1 | tee ../log/qwen-base/gsm8k-test-8shot-cot-qwen-1.5b.log
