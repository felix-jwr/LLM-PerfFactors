#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test_mmlu.py \
  --model_name "unsloth/gemma-2-9b-it-bnb-4bit" \
  --chat_template "gemma-2" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names gsm8k \
  --no_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/mmlu-gsm8k/gemma-2-9b-it/test-0shot-nocot-gemma-2-9b-it