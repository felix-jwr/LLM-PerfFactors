import os
import gc
import torch
import pynvml
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map


def check_vram_usage(plot=False):
    # Initialize NVIDIA Management Library
    pynvml.nvmlInit()
    
    # Get the VRAM usage for each GPU
    vram_usage = []
    device_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]

    for dh in device_handles:
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(dh)
        vram_usage.append((memory_info.used // 1024**2) / 1000)    

    if plot:
        # Plot the VRAM usage
        plt.figure(figsize=(5, 2))
        plt.ylim(0, np.max(vram_usage))
        plt.bar(range(0, pynvml.nvmlUnitGetDeviceCount), vram_usage, color='blue', label='Used')
        plt.show()

    print(f"VRAM Usage: {[str(i) + 'GB' for i in vram_usage]}")
    pynvml.nvmlShutdown()
    
    return vram_usage


def download_model():
    # Only required if the model is not already downloaded
    if not os.path.exists("llama-2-7b"):
        access_token = open("access_token.txt", "r").read()
        login(token=access_token)
        model_path = snapshot_download("meta-llama/Llama-2-7b-hf", local_dir="./llama-2-7b-hf")
    else:
        model_path = os.path.join(os.getcwd(), "llama-2-7b-hf")

    print(f"Downloaded model to: {model_path}")
    return model_path


def load_model(model_name="llama-2-7b-hf", plot=False):
    # Download the model
    model_path = download_model()

    # Setup base model settings
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

    print(f"Loaded model weights")
    check_vram_usage()

    return model_path, base_model, llama_tokenizer


def download_dataset(data_name="mlabonne/guanaco-llama2-1k"):
    # TODO: Change this to download the correct dataset
    training_data = load_dataset(data_name, split="train")
    print(f"Loaded dataset. Shape: {training_data.shape}")

    return training_data


def empty_cache():
    print("Freeing up VRAM")
    torch.cuda.empty_cache()
    gc.collect()

    # Check the VRAM usage after emptying the cache
    allocated = (torch.cuda.memory_allocated() / 1e9)
    reserved = (torch.cuda.memory_reserved() / 1e9)
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


# ==================================================================================================
# Check if the GPU is available and allocated
# print(f"GPU available: {torch.cuda.is_available()}, Number of GPUs: {torch.cuda.device_count()}")
# check_vram_usage(plot=False)

# Download and load the model
# load_model()

# Check the VRAM usage after loading the model
# check_vram_usage(plot=False)

# Download the dataset
# download_dataset(data_name="wikitext")

# Empty the cache
# empty_cache()
# check_vram_usage(plot=False)
# ==================================================================================================
