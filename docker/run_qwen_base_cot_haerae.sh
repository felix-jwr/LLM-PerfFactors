#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
# python3 -u test_mmlu.py \
#   --model_name "Qwen/Qwen2.5-1.5B" \
#   --chat_template "qwen" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_cot \
#   --use_cot \
#   --n_shot 0 \
#   --batch_size 32 \
#   2>&1 | tee ../log/qwen-base-haerae/test-haerae-0shot-cot-qwen1.5b.log

# python3 -u test_mmlu.py \
#   --model_name "Qwen/Qwen2.5-1.5B" \
#   --chat_template "qwen" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_cot \
#   --use_cot \
#   --n_shot 8 \
#   --batch_size 8 \
#   2>&1 | tee ../log/qwen-base-haerae/test-haerae-8shot-cot-qwen1.5b.log

# python3 -u test_mmlu.py \
#   --model_name "Qwen/Qwen2.5-7B" \
#   --chat_template "qwen" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_cot \
#   --use_cot \
#   --n_shot 0 \
#   --batch_size 8 \
#   2>&1 | tee ../log/qwen-base-haerae/test-haerae-0shot-cot-qwen7b.log

# python3 -u test_mmlu.py \
#   --model_name "Qwen/Qwen2.5-7B" \
#   --chat_template "qwen" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_cot \
#   --use_cot \
#   --n_shot 8 \
#   --batch_size 4 \
#   2>&1 | tee ../log/qwen-base-haerae/test-haerae-8shot-cot-qwen7b.log

# python3 -u test_mmlu.py \
#   --model_name "Qwen/Qwen2.5-14B" \
#   --chat_template "qwen" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_cot \
#   --use_cot \
#   --n_shot 0 \
#   --batch_size 4 \
#   2>&1 | tee ../log/qwen-base-haerae/test-haerae-0shot-cot-qwen14b.log

# python3 -u test_mmlu.py \
#   --model_name "Qwen/Qwen2.5-14B" \
#   --chat_template "qwen" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_cot \
#   --use_cot \
#   --n_shot 8 \
#   --batch_size 1 \
#   2>&1 | tee ../log/qwen-base-haerae/test-haerae-8shot-cot-qwen14b.log

# python3 -u test_mmlu.py \
#   --model_name "Qwen/Qwen2.5-32B" \
#   --chat_template "qwen" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --multi-gpu \
#   --random_seed 42 \
#   --task_names haerae_cot \
#   --use_cot \
#   --n_shot 0 \
#   --batch_size 16 \
#   2>&1 | tee ../log/qwen-base-haerae/test-haerae-0shot-cot-qwen32b.log

# python3 -u test_mmlu.py \
#   --model_name "Qwen/Qwen2.5-32B" \
#   --chat_template "qwen" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --multi-gpu \
#   --random_seed 42 \
#   --task_names haerae_cot \
#   --use_cot \
#   --n_shot 8 \
#   --batch_size 4 \
#   2>&1 | tee ../log/qwen-base-haerae/test-haerae-8shot-cot-qwen32b.log


python3 -u test_mmlu.py \
  --model_name "Qwen/Qwen2.5-32B" \
  --chat_template "qwen" \
  --max_seq_length 512 \
  --load_in_4bit \
  --multi-gpu \
  --random_seed 42 \
  --task_names gsm8k \
  --no_cot \
  --n_shot 0 \
  --batch_size 16 \
  2>&1 | tee ../log/test-gsm8k-0shot-nocot-qwen32b.log

python3 -u test_mmlu.py \
  --model_name "Qwen/Qwen2.5-32B" \
  --chat_template "qwen" \
  --max_seq_length 512 \
  --load_in_4bit \
  --multi-gpu \
  --random_seed 42 \
  --task_names gsm8k_cot \
  --use_cot \
  --n_shot 0 \
  --batch_size 16 \
  2>&1 | tee ../log/test-gsm8k-0shot-cot-qwen32b.log

python3 -u test_mmlu.py \
  --model_name "Qwen/Qwen2.5-32B" \
  --chat_template "qwen" \
  --max_seq_length 512 \
  --load_in_4bit \
  --multi-gpu \
  --random_seed 42 \
  --task_names gsm8k \
  --no_cot \
  --n_shot 8 \
  --batch_size 8 \
  2>&1 | tee ../log/test-gsm8k-8shot-nocot-qwen32b.log

python3 -u test_mmlu.py \
  --model_name "Qwen/Qwen2.5-32B" \
  --chat_template "qwen" \
  --max_seq_length 512 \
  --load_in_4bit \
  --multi-gpu \
  --random_seed 42 \
  --task_names gsm8k_cot \
  --use_cot \
  --n_shot 8 \
  --batch_size 8 \
  2>&1 | tee ../log/test-gsm8k-8shot-usecot-qwen32b.log


