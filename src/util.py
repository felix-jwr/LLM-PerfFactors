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


def check_vram_usage(plot=False):
    """
    Check the current VRAM usage using pynvml.

    return: List of VRAM usage for each GPU.
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

    model: Model to delete.
    trainer: Trainer to delete.
    return: None.
    """

    del model
    del trainer
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()