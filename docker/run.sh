#!/bin/bash

cd /workspace/src/

# python3 -u fine_tune.py 2>&1 | tee ../log/gsm8k/finetune/ft-phi-4-bnb-4bit

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --chat_template "llama-3.1" \
  --max_seq_length 512 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset "openai/gsm8k" \
  --subset "main" \
  --split "test" \
  --no_cot \
  --n_shot 0 \
  --batch_size 4 \
  2>&1 | tee ../log/test-0shot-nocot-meta-llama-3.1-8b-instruct-smartcode-new

# python3 -u temp.py 2>&1 | tee ../log/test-8shot-cot-meta-llama-3.1-8b-instruct-dumbcode-defunctioned
