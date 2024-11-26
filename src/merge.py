import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from peft import PeftModel
from util import empty_vram


def merge_weights_with_model(base_model_name="meta-llama/Llama-2-7b-chat-hf" , ft_weights_dir="../llama-2-7b-ft-weights",
                             device_map="auto", save=False, output_dir="../ft-model-merged"):
    """
    Merge LoRA weights with base model and save the merged model.

    Args: 
        model_name: Name of the base model to merge the weights with.
        device_map: Device map for the model (e.g. "auto", "cuda:0", etc.).
        output_dir: Directory to save the merged model.
    Returns: 
        str: Directory where the merged model is saved.
    """

    # Reload tokeniser
    tokeniser = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = "right"

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # Merge fine-tuned model with base model
    merged_model = PeftModel.from_pretrained(base_model, ft_weights_dir)
    merged_model = merged_model.merge_and_unload()

    if save:
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return merged_model, tokeniser, output_dir