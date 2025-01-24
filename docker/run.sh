#!/bin/bash

# Clean up the log and results directories
# rm -rf /app/log/*
# rm -rf /app/results/*

# Run the training and testing scripts
cd /app/src
# python3 -u train.py 2>&1 | tee ../results/train_output.log
python3 -u test_llama.py 2>&1 | tee ../log/test-llama-3.2-3B-Instruct-bnb-4bit