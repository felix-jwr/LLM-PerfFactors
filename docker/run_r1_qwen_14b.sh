#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test_gsm.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/distill/gsm8k-test-0shot-nocot-deepseek-qwen-14

python3 -u test_gsm.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --use_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/distill/gsm8k-test-0shot-cot-deepseek-qwen-14

python3 -u test_gsm.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --no_cot \
  --n_shot 8 \
  --batch_size 4 \
  2>&1 | tee ../log/distill/gsm8k-test-8shot-nocot-deepseek-qwen-14

python3 -u test_gsm.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --use_cot \
  --n_shot 8 \
  --batch_size 4 \
  2>&1 | tee ../log/distill/gsm8k-test-8shot-cot-deepseek-qwen-14

