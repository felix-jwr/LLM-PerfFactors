#!/bin/bash
cd /app/src
# python3 train.py
# python3 test.py
# python3 -u train.py 2>&1 | tee ../results/train_output.log
python3 -u test.py 2>&1 | tee ../results/test_output.log