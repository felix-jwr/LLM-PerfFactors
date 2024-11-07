import torch
# from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
    # HfArgumentParser,
    # TrainingArguments,
    # pipeline,
    # logging,
)
from datasets import load_dataset
# from huggingface_hub import login
from peft import (
    PeftModel,
    # LoraConfig,
)
from util import empty_vram


def merge_weights_with_model(model_name="meta-llama/Llama-2-7b-chat-hf" , device_map="auto", output_dir="../ft-model-merged"):
    """
    Merge LoRA weights with base model and save the merged model.

    Args: 
        model_name: Name of the base model to merge the weights with.
        device_map: Device map for the model (e.g. "auto", "cuda:0", etc.).
        output_dir: Directory to save the merged model.
    Returns: 
        str: Directory where the merged model is saved.
    """
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, output_dir)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir