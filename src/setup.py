import os
import gc
import torch
import pynvml
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM


def check_vram_usage(plot=False):
    # Initialize NVIDIA Management Library
    pynvml.nvmlInit()
    
    # Get the VRAM usage for each GPU
    device_count = pynvml.nvmlDeviceGetCount() 
    vram_usage = {}
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_usage[i] = {
            'total': memory_info.total,
            'used': memory_info.used,
            'free': memory_info.free
        }

    # Extract keys, total, and used values
    keys, total, used = zip(*[(i, vram_usage[i]['total'], vram_usage[i]['used']) for i in vram_usage.keys()])

    if plot:
        # Plot the VRAM usage
        plt.figure(figsize=(5, 2))
        plt.ylim(0, np.max(total))
        plt.bar(keys, used, color='blue', label='Used')
        plt.show()

    print(used)
    pynvml.nvmlShutdown()
    
    return vram_usage


def download_model():
    # Only required if the model is not already downloaded
    if not os.path.exists("llama-2-7b"):
        access_token = open("access_token.txt", "r").read()
        login(token=access_token)

    model_path = snapshot_download("meta-llama/Llama-2-7b-hf", local_dir="./llama-2-7b")

    return model_path


def load_model(model_name="llama-2-7b-hf"):
    base_model_name = model_name
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto"
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1


def download_dataset(data_name="wikitext"):
    # TODO: Change this to download the correct dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    return dataset


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()

    # Check the VRAM usage after emptying the cache
    allocated = (torch.cuda.memory_allocated() / 1e9)
    reserved = (torch.cuda.memory_reserved() / 1e9)
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# ==================================================================================================
# Check if the GPU is available and allocated
print(f"GPU available: {torch.cuda.is_available()}, Number of GPUs: {torch.cuda.device_count()}")
check_vram_usage(plot=False)

# Download and load the model
download_model()
load_model()

# Check the VRAM usage after loading the model
check_vram_usage(plot=False)

# Download the dataset
# download_dataset(data_name="wikitext")

# Empty the cache
empty_cache()
check_vram_usage(plot=False)
# ==================================================================================================
