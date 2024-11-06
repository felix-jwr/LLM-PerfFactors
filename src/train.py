import torch
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel
from util import empty_vram, check_vram_usage

# Login to HF
access_key = open('access_token.txt','r').read()
login(token = access_key)

# Set the name of the model to train, the dataset to use, and the name of the new (fine-tuned) model
model_name = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "llama-2-7b-ft-weights"

################################################################################
# QLoRA parameters
################################################################################

lora_r = 64                         # LoRA attention dimension
lora_alpha = 16                     # Alpha parameter for LoRA scaling
lora_dropout = 0.1                  # Dropout probability for LoRA layers

################################################################################
# bitsandbytes parameters
################################################################################

use_4bit = True                     # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"         # Quantization type (fp4 or nf4)
use_nested_quant = False            # Activate nested quantization for 4-bit base models (double quantization)

################################################################################
# TrainingArguments parameters
################################################################################


output_dir = "./results"            # Output directory where the model predictions and checkpoints will be stored
num_train_epochs = 1                # Number of training epochs
fp16 = False                        # Enable fp16/bf16 training (set bf16 to True with an A100)
bf16 = False
per_device_train_batch_size = 4     # Batch size per GPU for training
per_device_eval_batch_size = 4      # Batch size per GPU for evaluation
gradient_accumulation_steps = 1     # Number of update steps to accumulate the gradients for
gradient_checkpointing = True       # Enable gradient checkpointing
max_grad_norm = 0.3                 # Maximum gradient normal (gradient clipping)
learning_rate = 2e-4                # Initial learning rate (AdamW optimizer)
weight_decay = 0.001                # Weight decay to apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"         # Optimizer to use (AdamW, PagedAdamW, etc.)
lr_scheduler_type = "cosine"        # Learning rate schedule
max_steps = -1                      # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03                 # Ratio of steps for a linear warmup (from 0 to learning rate)
group_by_length = True              # Group sequences into batches with same length. Saves memory and speeds up training
save_steps = 0                      # Save checkpoint every X updates steps (0 to disable)
logging_steps = 25                  # Log every X updates steps (0 to disable)

################################################################################
# SFT parameters
################################################################################

max_seq_length = None               # Maximum sequence length to use
packing = False                     # Pack multiple short examples in the same input sequence to increase efficiency
device_map = "auto"                 # Where to load the model (auto, cuda:0, etc.)

################################################################################
# Load Dataset
################################################################################

# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning (SFT) parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Check VRAM usage after loading everything
check_vram_usage(plot=False)

# Train model and save
trainer.train()
trainer.model.save_pretrained(new_model)

# Empty VRAM
empty_vram(model=model, trainer=trainer)

################################################################################
# Merge LoRA weights with base model
################################################################################

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

merged_model_dir = "ft-model-merged"
model.save_pretrained(merged_model_dir)
tokenizer.save_pretrained(merged_model_dir)

# Empty VRAM
empty_vram(model=model, trainer=None)