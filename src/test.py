import os
import json
import time
import torch
import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from datasets import load_dataset
from huggingface_hub import login
from vllm import LLM, SamplingParams
from peft import LoraConfig, PeftModel
from util import empty_vram, check_vram_usage, evaluate_perplexity

################################################################################
# Dataset
################################################################################

# Login to HF
access_key = os.environ['API_TOKEN']
login(token = access_key)

# Set the name of the base model, the dataset to use
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
 
# Load the dataset
def auto_load_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split)
    return dataset
test_dataset = (auto_load_dataset("openai/gsm8k", "main"))["test"]

################################################################################
# Merged Model
################################################################################

# Name of fine-tuned model and the merged model
ft_model_dir = "../llama-2-7b-ft-weights"
merged_model_dir = "../llama-2-7b-ft-merged"

if not os.path.exists(ft_model_dir):
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Merge fine-tuned model with base model
    merged_model = PeftModel.from_pretrained(base_model, ft_model_dir)
    merged_model = merged_model.merge_and_unload()

    # Reload tokenizer to save it
    tokeniser = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = "right"

    # Save the merged model
    merged_model.save_pretrained(merged_model_dir)
    tokeniser.save_pretrained(merged_model_dir)

    # Get max token length
    max_tokens = max(len(tokeniser.encode(q + a)) for q, a in zip(test_dataset["question"], test_dataset["answer"]))
    print(f"\nMax tokens: {max_tokens}")

################################################################################
# Inference
################################################################################

def save_outputs(outputs, questions, answers, model_path):
    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the filename with the date, time, and model path
    filename = f"../results/{timestamp}_{model_path.replace('/', '_')}_outputs.json"
    
    # Create a list of dictionaries to store the results
    results = []
    for q, a, o in zip(questions, answers, outputs):
        results.append({
            "Question": q,
            "Answer": a,
            "Output": o.outputs[0].text
        })
    
    # Save the results to a JSON file
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Saved outputs to {filename}")

def run_inference(model_path, tokeniser_path, test_dataset):
    # Initialise VLLM with current model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        tokenizer=tokeniser_path,
        trust_remote_code=True,
    )

    # Set sampling parameters for generating responses
    sampling_parameters = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=None,
        stop=["</s>"],  # Stop token for llama models
        logprobs=True,
    )

    # Generate responses for each question
    print(f"\n{'#' * 80}\n{model_path} Inference\n{'#' * 80}\n")
    questions = test_dataset["question"]
    answers = test_dataset["answer"]
    outputs = llm.generate(questions, sampling_parameters)

    return outputs, questions, answers


# Run inference on base model
test_dataset = (auto_load_dataset("openai/gsm8k", "main"))["test"]
outputs, questions, answers = run_inference(base_model_name, base_model_name, test_dataset)
save_outputs(outputs, questions, answers, base_model_name)
time.sleep(5)   # Sleep to allow VRAM to empty
empty_vram()

# TODO: Clean up tokeniser to avoid crashing due to fork error

# Run inference on fine-tuned model
test_dataset = (auto_load_dataset("openai/gsm8k", "main"))["test"]
outputs, questions, answers = run_inference(merged_model_dir, merged_model_dir, test_dataset)
save_outputs(outputs, questions, answers, merged_model_dir)
time.sleep(5)
empty_vram()