#!/bin/bash

# UNCOMMENT AFTER MISTRAL 7B COT WITH TEMPLATE FINISHES
# mkdir -p /app/results/gsm8k
# find /app/results -mindepth 1 -maxdepth 1 -type d -exec mv {} /app/results/gsm8k/ \;

# mkdir -p /app/log/gsm8k
# find /app/log -mindepth 1 -maxdepth 1 -type d -exec mv {} /app/log/gsm8k/ \;

# Run the training and testing scripts
cd /app/src
python3 -u fine_tune.py 2>&1 | tee ../log/gsm8k/finetune/ft-llama-3.1-8B-Instruct-bnb-4bit
# python3 -u test_llama.py 2>&1 | tee ../log/gsm8k/test/test-llama-3.1-70B-Instruct-bnb-4bit-NoCoT