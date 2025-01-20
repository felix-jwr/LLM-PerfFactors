import gc
import re
import os
import json
import torch
import random
import pynvml
import datetime
import numpy as np
import matplotlib.pyplot as plt


def check_vram_usage(plot: bool = False) -> list:
    """
    Check the current VRAM usage using pynvml.

    args:
        plot: bool, Whether to plot the VRAM usage.

    returns:
        vram_usage: list, VRAM usage for each GPU.
    """

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

    print(f'VRAM Usage: {[str(i) + "GB" for i in vram_usage]}')
    pynvml.nvmlShutdown()

    return vram_usage


def empty_vram(model = None, trainer = None) -> None:
    """
    Empty the VRAM by deleting all variables and running garbage collection.

    args:
        model: (any), Model to delete.
        trainer: (any), Trainer to delete.

    returns:
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


def extract_answer(text: str, eos: str = None, truth: bool = False) -> str:
    """
    Extract the last numerical answer from a piece of text using regular expressions.

    args:
        text: str, Example to process.
        eos: str, End of string delimiter.
        truth: bool, Whether the text is a ground truth answer.

    returns:
        output: str, The extracted numerical output from the text.
    """

    if truth:
        answer = re.split(r'####', text)[-1].strip()
        output = re.sub(r'[,\$£€¥%g]', '', answer)
        return output

    # If eos is provided, split on it first
    if eos:
        text = re.split(re.escape(eos), text)[0].strip()
    
    # Look for numbers with optional decimal points, commas, and currency symbols
    all_numbers = re.findall(r'[,\$£€¥%g]?(\d+(?:,\d+)*(?:\.\d+)?)', text)
    
    if all_numbers:
        answer = all_numbers[-1].strip()
        output = re.sub(r'[,\$£€¥%g]', '', answer)#
        return output
    
    return ''


def save_results(model_name: str, dataset_name: str, n_shot: int, results: list) -> None:
    """
    Save a results .json from a model evaluation.

    args:
        model_name: str, The model name. Should ideally be the short name (i.e. excluding 'unsloth/').
        dataset_name: str, The name of the dataset, 
        n_shot: int, The number of example questions used for the n-shot test.
        results: list, The results of model evaluation.

    returns:
        None
    """

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Clear up any slashes in the model name to avoid making directories
    model_name = model_name.replace('/', '_')
    dataset_name = dataset_name.replace('/', '_')

    os.makedirs(f'../results/{model_name}', exist_ok=True)
    result_file = f'../results/{model_name}/{dataset_name}_{n_shot}-shot_{timestamp}.json'

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Results saved to {result_file}')


def generate_n_shot_prompt(n_shot_data: dict, n: int, question: str, seed: int) -> str:
    """
    Generate a prompt for the model with n example questions for an n-shot prompt.

    args:
        n_shot_data: dict, Training examples to use as n shots. Mustn't be from the test set.
        n: int, The number of example questions to include. Must be > 0.
        question: str, The actual prompt from the test set.
        seed: int, The random seed to use for reproducibility.

    returns:
        string: str, The n-shot prompt for the model.
    """

    def question_prompt(string):
        return f"Q: {string}"

    def answer_prompt(string):
        return f"A: {string}"

    prompts = []

    random.seed(seed)
    for question_and_answer in random.sample(n_shot_data, n):
        prompts.append({"role": "user", "content": question_prompt(question_and_answer["question"])})
        prompts.append({"role": "assistant", "content": answer_prompt(question_and_answer["answer"])})

    # TODO: Refactor this to use flag
    # CoT Prompt
    prompts.append({"role": "user", "content": question_prompt(question) + " Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})
    # No CoT
    # prompts.append({"role": "user", "content": question_prompt(question) + " You MUST write the answer as an integer after '####'."})

    return prompts