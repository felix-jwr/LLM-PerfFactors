#!/bin/bash

cd /workspace/src/

# NOTE: Change --no_cot to --use_cot to enable COT, similarly for --load_in_4bit

##### Randomness test #####
python3 -u test_gsm.py \
--model_name "meta-llama/Llama-3.1-8B-Instruct" \
--chat_template "llama-3.1" \
--max_seq_length 512 \
--load_in_4bit \
--random_seed 900 \
--dataset openai/gsm8k \
--no_cot \
--n_shot 0 \
--batch_size 16 \
2>&1 | tee ../log/test-gsm8k-0shot-nocot-Llama-3.1-8B-Instruct-SEED-900.log

python3 -u test_gsm.py \
--model_name "meta-llama/Llama-3.1-8B-Instruct" \
--chat_template "llama-3.1" \
--max_seq_length 512 \
--load_in_4bit \
--random_seed 3141592784 \
--dataset openai/gsm8k \
--no_cot \
--n_shot 0 \
--batch_size 16 \
2>&1 | tee ../log/test-gsm8k-0shot-nocot-Llama-3.1-8B-Instruct-SEED-3141592784.log
