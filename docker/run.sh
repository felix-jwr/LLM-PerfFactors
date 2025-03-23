#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
python3 -u test_mmlu.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct" \
  --chat_template "llama" \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae \
  --no_cot \
  --n_shot 0 \
  --batch_size 16 \
  2>&1 | tee ../log/mmlu-haerae/llama-3.2-3b-instruct/test-0shot-nocot-llama-3.2-3b-instruct