#!/bin/bash

# Run the training and testing scripts
cd /app/src
# python3 -u fine_tune.py 2>&1 | tee ../log/gsm8k/finetune/ft-phi-4-bnb-4bit
python3 -u single_gpu_test.py 2>&1 | tee ../log/gsm8k/test/test-llama-3.2-3B-Instruct-bnb-4bit-CoT