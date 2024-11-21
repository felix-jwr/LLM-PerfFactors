import os
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

################################################################################
# Dataset
################################################################################

# Login to HF
access_key = os.environ['API_TOKEN']
login(token = access_key)

# Set the name of the model to train, the dataset to use, and the name of the new (fine-tuned) model
model_name = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "openai/gsm8k"
new_model = "../llama-2-7b-ft-weights"

# Load dataset
dataset = load_dataset(dataset_name, "main")
train_dataset = dataset["train"]

################################################################################
#  Tokenisation
################################################################################

# Load tokeniser and model with QLoRA configuration
use_4bit = True                     # Activate 4-bit precision base model loading
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,  # Activate 4-bit precision base model loading
    bnb_4bit_quant_type="nf4",  # Quantisation type (fp4 or nf4)
    bnb_4bit_compute_dtype="float16",   # Compute dtype for 4-bit base models
    bnb_4bit_use_double_quant=False, # Activate nested quantisation for 4-bit base models (double quant)
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
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokeniser
tokeniser = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokeniser.pad_token = tokeniser.eos_token
tokeniser.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,  # Alpha parameter for LoRA scaling
    lora_dropout=0.1,  # Dropout probability for LoRA layers
    r=64,   # LoRA attention dimension
    bias="none",
    task_type="CAUSAL_LM",
)

# Get max token length
max_tokens = max(len(tokeniser.encode(q + a)) for q, a in zip(train_dataset["question"], train_dataset["answer"]))
print(f"\nMax tokens: {max_tokens}")

################################################################################
# Training
################################################################################

# Set training parameters
training_arguments = TrainingArguments(
    output_dir="../results",    # Output directory where the model predictions and checkpoints will be stored
    num_train_epochs=1, # Number of training epochs
    per_device_train_batch_size=4,    # Batch size per GPU for training
    gradient_accumulation_steps=4,    # Batch size per GPU for evaluation
    optim="paged_adamw_32bit",  # Optimizer to use (AdamW, PagedAdamW, etc.)
    save_steps=0,  # Save checkpoint every X updates steps (0 to disable)
    logging_steps=25,   # Log every X updates steps (0 to disable)
    learning_rate=2e-4, # Initial learning rate (AdamW optimizer)
    weight_decay=0.001, # Weight decay to apply to all layers except bias/LayerNorm weights
    fp16=False,  # Enable fp16 training
    bf16=False,  # Enably bf16 training (set to True with an A100)
    max_grad_norm=0.3,  # Maximum gradient normal (gradient clipping)
    max_steps=-1,    # Number of training steps (overrides num_train_epochs)
    warmup_ratio=0.03,  # Ratio of steps for a linear warmup (from 0 to learning rate)
    group_by_length=False,    # Group sequences into batches with same length. Saves mem and speeds up train
    lr_scheduler_type="cosine", # Learning rate schedule
    report_to="tensorboard"
)

# Set supervised fine-tuning (SFT) parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="answer",    # Only tokenise the answer text, and predict only on answer tokens
    max_seq_length=max_tokens,
    tokenizer=tokeniser,
    args=training_arguments,
    packing=True,
)

# Check VRAM usage after loading everything
check_vram_usage(plot=False)

# Train model and save
trainer.train()
trainer.model.save_pretrained(new_model)
tokeniser.save_pretrained(new_model)

# Empty VRAM
empty_vram(model=model, trainer=trainer)