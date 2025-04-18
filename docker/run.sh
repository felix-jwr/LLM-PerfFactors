#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test_gsm.py \
  --model_name "Qwen/Qwen2.5-7B" \
  --chat_template "qwen" \
  --load_in_4bit \
  --max_seq_length 512 \
  --random_seed 42 \
  --dataset openai/gsm8k \
  --subset main \
  --split test \
  --use_cot \
  --n_shot 8 \
  --batch_size 8 \
  2>&1 | tee ../log/qwen-base/gsm8k-test-8shot-cot-qwen-7b.log
