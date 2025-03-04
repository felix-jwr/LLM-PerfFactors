#!/bin/bash

cd /workspace/src/

# python3 -u fine_tune.py 2>&1 | tee ../log/gsm8k/finetune/ft-phi-4-bnb-4bit

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u single_gpu_test.py \
  --model_name "unsloth/Llama-3.1-8B-Instruct-bnb-4bit" \
  --chat_template "llama-3.1" \
  --max_seq_length 2048 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset "openai/gsm8k" \
  --subset "main" \
  --split "test" \
  --no_cot \
  --n_shot 8 \
  2>&1 | tee ../log/test5-8shot-nocot-llama-3.1-8b-instruct

  python3 -u single_gpu_test.py \
  --model_name "unsloth/Llama-3.1-8B-Instruct-bnb-4bit" \
  --chat_template "llama-3.1" \
  --max_seq_length 2048 \
  --load_in_4bit \
  --random_seed 42 \
  --dataset "openai/gsm8k" \
  --subset "main" \
  --split "test" \
  --use_cot \
  --n_shot 8 \
  2>&1 | tee ../log/test5-8shot-cot-llama-3.1-8b-instruct
