#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test_mmlu.py \
  --model_name "unsloth/gemma-2-2b-it-bnb-4bit" \
  --chat_template "gemma" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset "openai/gsm8k" \
  --subset "main" \
  --split "test" \
  --no_cot \
  --n_shot 8 \
  --batch_size 1 \
  2>&1 | tee ../log/mmlu/test-import-lm-eval