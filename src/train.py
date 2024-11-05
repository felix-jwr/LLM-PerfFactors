import os
import torch
import pynvml
import numpy as np

from trl import SFTTrainer
from typing import List, Optional
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import init_empty_weights, infer_auto_device_map
from setup import load_model, download_dataset, check_vram_usage, empty_cache
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline


# Load model and dataset
print(f"GPU available: {torch.cuda.is_available()}, Number of GPUs: {torch.cuda.device_count()}")
model_path, base_model, llama_tokenizer = load_model(model_name="llama-2-7b-hf", plot=False)
training_data = download_dataset(data_name="mlabonne/guanaco-llama2-1k")
check_vram_usage()

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# LoRA Config
# reduce rank r if you're running out of vram
peft_parameters = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_parameters)
model.print_trainable_parameters()

# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained("llama-2-7b-enhanced")
check_vram_usage()
empty_cache()

# ==================================================================================================
# Ideally this stuff should move at some point, just testing it works first.

# Reload model in FP16 and merge it with LoRA weights
# make sure that both models are available before running or else inference will not work
base_model_name = "llama-2-7b-hf"
new_model_name = "llama-2-7b-enhanced"


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, new_model_name)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

output_dir = "llama-2-7b-merged"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ==================================================================================================
# This stuff should also move, this is for testing
# getting the perplexity without vllm
# seems right now that sampling params from vllm doesn't support logprobs for the prompt given

# getting pp
def calculate_perplexity(model, tokenizer, text, max_length=300):
    """
    Calculate the perplexity of a text using a language model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        max_length: Maximum sequence length to process
        
    Returns:
        float: The perplexity score
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

def evaluate_dataset(model, tokenizer, texts):
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
            ppl_and_loss = calculate_perplexity(model, tokenizer, text)
            ppl = ppl_and_loss[0]
            loss = ppl_and_loss[1]

            perplexities.append(ppl)
            total_loss.append(loss)

        except Exception as e:
            print(f"Error processing text: {e}")
            continue
    
    return perplexities, total_loss

model_path = "llama-2-7b-merged"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    # need to 
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

test_set = [
    "<s>[INST] Hello, how are you? [/INST] I'm doing fine thank you, I hope you're doing well too!</s>:",
    "<s>[INST] Are you real?[/INST] I'm a model made out of billions of parameters so that I can form sentences like this.</s>:",
]

ppl_and_loss = evaluate_dataset(model, tokenizer, test_set)
ppl = ppl_and_loss[0]
loss = ppl_and_loss[1]

print("Average perplexity: ", np.mean(ppl))
print(ppl)
print("Average loss: ", np.mean(loss))
print(loss)

## WRITING TO FILE OUTSIDE THE CONTAINER (NEEDS VOLUME TO BE MOUNTED)
# Create the output directory if it doesn't exist
out_dir = os.path.join(os.getcwd(), 'out')
os.makedirs(out_dir, exist_ok=True)

# Write the data to a file
out_file = os.path.join(out_dir, 'metrics.txt')
with open(out_file, 'w') as f:
    f.write(f"Average perplexity: {np.mean(ppl)}\n")
    f.write(f"{ppl}\n\n")
    f.write(f"Average loss: {np.mean(loss)}\n")
    f.write(f"{loss}\n")

print(f"Data written to {out_file}")
## END OF WRITING SECTION

empty_cache()
check_vram_usage()

# ==================================================================================================
# Can also put inference down here