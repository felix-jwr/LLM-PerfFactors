#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test_gsm.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset gsm8k \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/gsm8k-deepseek/test-gsm8k-0shot-nocot-Distill-Qwen-32B.log

python3 -u test_gsm.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset gsm8k \
  --subset main \
  --split test \
  --use_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/gsm8k-deepseek/test-gsm8k-0shot-cot-Distill-Qwen-32B.log

python3 -u test_gsm.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset gsm8k \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 8 \
  --batch_size 2 \
  2>&1 | tee ../log/gsm8k-deepseek/test-gsm8k-8shot-nocot-Distill-Qwen-32B.log

python3 -u test_gsm.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset gsm8k \
  --subset main \
  --split test \
  --use_cot \
  --n_shot 8 \
  --batch_size 2 \
  2>&1 | tee ../log/gsm8k-deepseek/test-gsm8k-8shot-cot-Distill-Qwen-32B.log