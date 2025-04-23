#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit
# python3 -u test_mmlu.py \
#   --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
#   --chat_template "deepseek" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_deepseek \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 64 \
#   2>&1 | tee ../log/haerae-deepseek/test-haerae-0shot-nocot-Distill-Qwen-1.5B.log

# python3 -u test_mmlu.py \
#   --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
#   --chat_template "deepseek" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_deepseek_cot \
#   --use_cot \
#   --n_shot 0 \
#   --batch_size 64 \
#   2>&1 | tee ../log/haerae-deepseek/test-haerae-0shot-cot-Distill-Qwen-1.5B.log

# python3 -u test_mmlu.py \
#   --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
#   --chat_template "deepseek" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_deepseek \
#   --no_cot \
#   --n_shot 8 \
#   --batch_size 16 \
#   2>&1 | tee ../log/haerae-deepseek/test-haerae-8shot-nocot-Distill-Qwen-1.5B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek_cot \
  --use_cot \
  --n_shot 8 \
  --batch_size 16 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-8shot-cot-Distill-Qwen-1.5B.log

# python3 -u test_mmlu.py \
#   --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
#   --chat_template "deepseek" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_deepseek \
#   --no_cot \
#   --n_shot 0 \
#   --batch_size 16 \
#   2>&1 | tee ../log/haerae-deepseek/test-haerae-0shot-nocot-Distill-Llama-8B.log

# python3 -u test_mmlu.py \
#   --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
#   --chat_template "deepseek" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_deepseek_cot \
#   --use_cot \
#   --n_shot 0 \
#   --batch_size 16 \
#   2>&1 | tee ../log/haerae-deepseek/test-haerae-0shot-cot-Distill-Llama-8B.log

# python3 -u test_mmlu.py \
#   --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
#   --chat_template "deepseek" \
#   --max_seq_length 512 \
#   --load_in_4bit \
#   --single_gpu \
#   --random_seed 42 \
#   --task_names haerae_deepseek \
#   --no_cot \
#   --n_shot 8 \
#   --batch_size 8 \
#   2>&1 | tee ../log/haerae-deepseek/test-haerae-8shot-nocot-Distill-Llama-8B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek_cot \
  --use_cot \
  --n_shot 8 \
  --batch_size 8 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-8shot-cot-Distill-Llama-8B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek \
  --no_cot \
  --n_shot 0 \
  --batch_size 16 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-0shot-nocot-Distill-Qwen-7B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek_cot \
  --use_cot \
  --n_shot 0 \
  --batch_size 16 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-0shot-cot-Distill-Qwen-7B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek \
  --no_cot \
  --n_shot 8 \
  --batch_size 8 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-8shot-nocot-Distill-Qwen-7B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek_cot \
  --use_cot \
  --n_shot 8 \
  --batch_size 8 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-8shot-cot-Distill-Qwen-7B.log


python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek \
  --no_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-0shot-nocot-Distill-Qwen-14B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek_cot \
  --use_cot \
  --n_shot 0 \
  --batch_size 8 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-0shot-cot-Distill-Qwen-14B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek \
  --no_cot \
  --n_shot 8 \
  --batch_size 4 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-8shot-nocot-Distill-Qwen-14B.log

python3 -u test_mmlu.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --chat_template "deepseek" \
  --max_seq_length 512 \
  --load_in_4bit \
  --single_gpu \
  --random_seed 42 \
  --task_names haerae_deepseek_cot \
  --use_cot \
  --n_shot 8 \
  --batch_size 4 \
  2>&1 | tee ../log/haerae-deepseek/test-haerae-8shot-cot-Distill-Qwen-14B.log