#!/bin/bash

# Run the training and testing scripts
cd /app/src
python3 -u fine_tune.py 2>&1 | tee ../log/gsm8k/finetune/ft-Mistral-Small-Instruct-2409-bnb-4bit
# python3 -u test_llama.py 2>&1 | tee ../log/gsm8k/test/test-llama-3.1-70B-Instruct-bnb-4bit-NoCoT