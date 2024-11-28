"""
A small collection of utility/helper functions.
Mostly useful for VRAM management, logging, and file I/O.
"""

import gc
import re
import os
import time
import torch
import random
import pynvml
import datetime
import numpy as np
import matplotlib.pyplot as plt

####################################################################################################
# Memory Management
####################################################################################################

def check_vram_usage(plot=False):
    """
    Check the current VRAM usage using pynvml.

    Args:
        plot: Whether to plot the VRAM usage.
        return: List of VRAM usage for each GPU.

    Returns:
        List: VRAM usage for each GPU.
    """

    # Initialize NVIDIA Management Library
    pynvml.nvmlInit()               

    # Get the VRAM usage for each GPU
    vram_usage = []
    device_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]

    for dh in device_handles:
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(dh)
        vram_usage.append((memory_info.used // 1024**2) / 1000)   # Convert to GB

    # Plot the VRAM usage (if requested)
    if plot:
        plt.figure(figsize=(5, 2))
        plt.ylim(0, np.max(vram_usage))
        plt.bar(range(0, pynvml.nvmlUnitGetDeviceCount), vram_usage, color='blue', label='Used')
        plt.show()

    print(f"VRAM Usage: {[str(i) + 'GB' for i in vram_usage]}")
    pynvml.nvmlShutdown()

    return vram_usage


def empty_vram(model=None, trainer=None):
    """
    Empty the VRAM by deleting all variables and running garbage collection.

    Args:
        model: Model to delete.
        trainer: Trainer to delete.

    Returns:
        None
    """

    try:
        del model
        del trainer
    except:
        pass

    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()

####################################################################################################
# Extract results
####################################################################################################

def extract_answer(text, eos=None):
    """
    Extract the model prediction generated from a GSM8k example using regular expressions.

    Args:
        text: Example to process.

    Returns:
        string: The extracted numerical output from the text.
    """
    if eos:
        text = re.split(re.escape(eos), text)[0].strip()

    text = re.split(r"####", text)[-1].strip()
    text = re.sub(r"[,\$%g]", "", text)

    return text

def save_results(results, model_name, dataset_name, n_shot):
    """
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = model_name.replace("/", "_")

    os.makedirs(f"../results/{model_name}", exist_ok=True)
    result_file = f"../results/{model_name}/{dataset_name}_{n_shot}-shot_{timestamp}_RESULTS.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")

####################################################################################################
# Dataset Specific
####################################################################################################

def generate_n_shot_prompt(n_shot_data, n, question):
    """
    """

    def question_prompt(string):
        return f"Q: {string}"

    def answer_prompt(string):
        return f"A: {string}"

    prompts = []

    random.seed(42)
    for question_and_answer in random.sample(n_shot_data, n):
        prompts.append({"role": "user", "content": question_prompt(question_and_answer["question"])})
        prompts.append({"role": "assistant", "content": answer_prompt(question_and_answer["answer"])})

    prompts.append({"role": "user", "content": question_prompt(question) + " Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})

    return prompts