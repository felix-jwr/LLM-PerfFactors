"""
A small collection of utility/helper functions.
Mostly useful for VRAM management, logging, and file I/O.
"""

import gc
import re
import os
import torch
import pynvml
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from transformers import StoppingCriteria

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

def extract_ground_truth_gsm8k(text):
    """
    Extract the ground truth from a GSM8k example.

    Args:
        text: Example to process.

    Returns:
        float: The ground truth from the text.
    """
    return text.split("####")[-1].strip()

def extract_predicted_gsm8k(text):
    """
    Extract the model prediction generated from a GSM8k example using regular expressions.

    Args:
        text: Example to process.

    Returns:
        float: The extracted numerical output from the text.
    """

    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)

    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        return None

####################################################################################################
# Generation
####################################################################################################

class SpecificStringStoppingCriteria(StoppingCriteria):
    """
    Stopping condition for generation on GSM8k. 
    Taken from: https://github.com/tianlwang/eval_gsm8k/blob/main/utils.py

    Args:
        tokenizer: Tokeniser (usually AutoTokenizer).
        stop_strings: List of stop strings to try.
        input_len: Length of a given input.

    Returns:
        str: Stop string for the current text.
    """
    def __init__(self, tokenizer, stop_strings, input_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[self.input_len:]
        
        return any(stop_string in current_text for stop_string in self.stop_strings)


def generate_model_answer(model, tokeniser, input_text, stopping_criteria_list, max_new_tokens=512):
    """
    Generate model answer for a given input text.
    
    Args:
        model: The language model.
        tokeniser: Tokeniser for the model.
        input_text: Input prompt string.
        stopping_criteria_list: Custom stopping criteria.
        max_new_tokens: Maximum number of new tokens to generate. Defaults to 512.
    
    Returns:
        dict: A dictionary containing the full output text and extracted numeric answer.
    """
    inputs = tokeniser(input_text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            pad_token_id=tokeniser.eos_token_id, 
            stopping_criteria=stopping_criteria_list
        )
    
    # Extract the final answer from the model's output
    output_text = tokeniser.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text.split("A:")[-1].strip()
    
    # Get the number out of the answer
    model_answer = extract_predicted_gsm8k(output_text)
    
    return {'text': output_text, 'numeric': model_answer}


def process_model_answers(model_answers):
    """
    Process model answers to get majority answer.
    
    Args:
        model_answers: List of model answer dictionaries.
    
    Returns:
        tuple: Majority answer and filtered numeric answers.
    """
    numeric_answers = [ma['numeric'] for ma in model_answers]
    filtered_answers = [num for num in numeric_answers if num is not None]
    majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None
    
    return majority_answer, numeric_answers

####################################################################################################
# Metrics
####################################################################################################

def get_single_perplexity(model, tokenizer, text, max_length=300):
    """
    Calculate the perplexity of a text using a language model.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        text: Input text to evaluate.
        max_length: Maximum sequence length to process.
        
    Returns:
        float: The perplexity score.
    """

    # Encode the text
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Get input IDs and create target labels (shifted by 1)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    
    # Calculate loss with no gradient tracking
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    
    # Calculate perplexity
    ppl = torch.exp(neg_log_likelihood)
    loss = neg_log_likelihood

    return ppl.item(), loss


def evaluate_perplexity(model, tokenizer, texts):
    """
    Calculate average perplexity across multiple texts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts to evaluate
        
    Returns:
        float: Average perplexity across all texts
    """

    perplexities = []
    total_loss = []

    for text in texts:
        try:
            perplexity, loss = get_single_perplexity(model, tokenizer, text)
            perplexities.append(perplexity)
            total_loss.append(loss)

        except Exception as e:
            print(f"Error processing text: {e}")
            continue
    
    return perplexities, total_loss