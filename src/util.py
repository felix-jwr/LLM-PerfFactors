"""
A small collection of utility/helper functions.
Mostly useful for VRAM management, logging, and file I/O.
"""

import gc
import os
import torch
import pynvml
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