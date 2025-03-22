#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test_mmlu.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct" \
  --chat_template "llama" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names gsm8k \
  --no_cot \
  --n_shot 8 \
  --batch_size 8 \
  2>&1 | tee ../log/mmlu-gsm8k/llama-3.2-3b-instruct/test-8shot-nocot-llama-3.2-3b-instruct
    
# python3 -u test_mmlu.py \
#   --model_name "meta-llama/Llama-3.1-8B-Instruct" \
#   --chat_template "llama" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names gsm8k \
#   --no_cot \
#   --n_shot 8 \
#   --batch_size 8 \
#   2>&1 | tee ../log/mmlu-gsm8k/llama-3.1-8b-instruct/test-8shot-nocot-llama-3.1-8b-instruct