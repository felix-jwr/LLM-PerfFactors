#!/bin/bash

cd /workspace/src/

# python3 -u fine_tune.py 2>&1 | tee ../log/gsm8k/finetune/ft-phi-4-bnb-4bit

# NOTE: Change --no_cot to --cot to enable COT, similarly for --load_in_4bit
python3 -u single_gpu_test.py \
  --model_name "unsloth/Llama-3.1-8B-Instruct-bnb-4bit" \
  --chat_template "unsloth" \
  --max_seq_length 2048 \
  --load_in_4bit \
  --dataset "openai/gsm8k" \
  --subset "main" \
  --split "test" \
  --no_cot \
  --n_shot 0 \
  2>&1 | tee ../log/temp-docker-local-setup