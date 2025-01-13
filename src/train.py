import os
import torch
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from util import empty_vram, check_vram_usage


# Set the random seed for reproducibility
random_seed=42
torch.manual_seed(random_seed)

################################################################################
#  Model and Tokeniser
################################################################################

# Set the name of the model to train, and the name of the new (fine-tuned) model
model_name = "unsloth/Meta-Llama-3.1-8B"
new_model = "../3.1-8B-gsm8k-ft-weights"
dtype = None    # Use None for auto-detection
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokeniser = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,   
    load_in_4bit = load_in_4bit,
    token = os.environ['API_TOKEN'], # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context (uses 30% less VRAM, fits 2x larger batch sizes!)
    random_state = random_seed,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

################################################################################
# Data Prep
################################################################################

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokeniser.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["question"]  # NOTE: This will need changing depending on the dataset.
    outputs      = examples["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# Login to HF
access_key = os.environ['API_TOKEN']
login(token = access_key)

# Load dataset
dataset_name = "openai/gsm8k"
dataset = load_dataset(dataset_name, "main", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,) # Format the prompts using the above function

# Legacy code for loading a subset of the dataset
# # Shuffle and take a subset (e.g., 10% of the original dataset)
# # Cuts down dataset from 676k rows to 67.6k, saves time for trainig proof of concept
# all_data = dataset["train"]
# subset_size = int(0.5 * len(all_data))  # Adjust the fraction as needed
# small_all_data = all_data.shuffle(seed=random_seed).select(range(subset_size))

# # Extra for code dataset - make a train test split
# split_dataset = small_all_data.train_test_split(test_size=0.2, seed=random_seed)
# train_dataset = split_dataset["train"]
# test_dataset = split_dataset["test"]
# print("\nLoaded dataset.")

################################################################################
# Training
################################################################################

# Set training parameters
training_arguments = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 16,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 8500, # Number of training steps (overrides num_train_epochs)
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True,
        logging_steps = 25,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = random_seed,
        output_dir = f"../results/{model_name.replace('/', '_')}",
        report_to = "tensorboard", # Use this for WandB etc
    )

# Set supervised fine-tuning (SFT) parameters
trainer = SFTTrainer(
    model = model,
    tokenizer = tokeniser,
    train_dataset = dataset,
    dataset_text_field = "text", # TODO: Data processing
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_arguments,
)

# Check VRAM usage after loading everything
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train model and save
print("\nStarting fine tuning...")
trainer_stats = trainer.train()
print("\nFine tuning complete!")
trainer.model.save_pretrained(new_model)
tokeniser.save_pretrained(new_model)

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Empty VRAM
empty_vram(model=model, trainer=trainer)