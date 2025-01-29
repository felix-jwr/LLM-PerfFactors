from trl import SFTTrainer
from peft import LoraConfig
from huggingface_hub import login
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from util import empty_vram, check_vram_usage
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


RANDOM_STATE = 42
HF_TOKEN = open('./hf_token.txt', 'r').read().strip()


def init(model_name: str, dtype: str, load_in_4_bit: bool) -> tuple:
    """
    Initialise Model and Tokeniser.

    args:
        model_name: str, Name of the model to load from HF
        dtype: str, Data type for the model
        load_in_4_bit: bool, Whether to load the model in 4-bit precision

    returns:
        model: AutoModelForCausalLM, The loaded HF model
        tokeniser: AutoTokenizer, The loaded HF tokeniser
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = dtype,
        device_map = 'auto',
        trust_remote_code = True
    )

    tokeniser = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True
    )
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = 'right'

    return model, tokeniser


def format_dataset(dataset_name: str, subset_name: str, split_name: str, tokeniser) -> tuple:
    """
    Load a dataset form HF, and apply preprocessing to pair instructions with responses for fine-tuning.

    args:
        dataset_name: str, The name of the dataset to load from HF.
        subset_name: str, The name of the subset to load.
        split_name: str, The name of the split to load.
        tokeniser: AutoTokenizer, The tokeniser to use for formatting.

    returns:
        dataset: dict, The loaded dataset.
    """

    # Set up formatting for prompts and responses to fine tune the model
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokeniser.eos_token
    def format_prompt(examples):
        instructions = examples['question']  # NOTE: These fields will need changing depending on the dataset.
        outputs = examples['answer']
        texts = []

        for instruction, output in zip(instructions, outputs):
            # Must add EOS_TOKEN, otherwise generation will go on forever
            text = prompt.format(instruction, output) + EOS_TOKEN
            texts.append(text)

        return { 'text' : texts, }

    # Load the dataset from HF, apply formatting
    dataset = load_dataset(dataset_name, subset_name, split = split_name)
    dataset = dataset.map(format_prompt, batched = True)
    
    return dataset

def fine_tune_model(model: AutoModelForCausalLM, tokeniser: AutoTokenizer, dataset: dict, max_seq_length: int,
                    batch_size_per_gpu: int, model_name: str, dataset_name: str) -> SFTTrainer:
    """
    Fine tunes a model on a given dataset using TRL's SFTTrainer.

    args:
        model: AutoModelForCausalLM, The model to fine-tune.
        tokeniser: AutoTokenizer, The tokeniser for the model.
        dataset: dict, The dataset to fine-tune on.
        max_seq_length: int, Maximum sequence length.
        batch_size_per_gpu: int, Batch size per GPU.
        model_name: str, The name of the model, used for saving the fine-tuned weights.
        dataset_name: str, The name of the dataset, used for saving the fine-tuned weights.

    returns:
        trainer: SFTTrainer, The fine-tuning trainer.
    """

    # Set up the LoRA config
    lora_config = LoraConfig(
        r = 16,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = 'none',
        task_type = 'CAUSAL_LM'
    )

    # Fine-tune the model
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokeniser,
        train_dataset = dataset,
        dataset_text_field = 'text',
        peft_config = lora_config,
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,                                   # Number of processes to use for data loading
        packing = False,                                        # Can make training 5x faster for short sequences
        args = TrainingArguments(
            num_train_epochs = 1,                               # Number of epochs to train for, overridden by max_steps
            per_device_train_batch_size = batch_size_per_gpu,   # Batch size per GPU
            gradient_accumulation_steps = 4,                    # Number of steps before backprop
            warmup_steps = 5,
            # max_steps = 3,                                    # Number of training steps, overrides num_train_epochs
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 250,                                # Log updates every n steps, set to 0 to disable
            optim = 'adamw_8bit',
            weight_decay = 0.01,
            lr_scheduler_type = 'linear',
            seed = RANDOM_STATE,
            output_dir = f'../ft_outputs/{dataset_name}/{model_name}/checkpoints',
            report_to = 'none',                                 # Use this for tensorboard, wandb, etc.
        ),
    )

    # Only saves the fine-tuned (LoRA) weights
    print('Starting fine tuning.')
    trainer.train()
    trainer.model.save_pretrained(f'../ft_outputs/{dataset_name}/{model_name}/weights')

    return trainer


if __name__ == '__main__':
    #################################
    #            SETTINGS           #
    #################################
    
    # Model settings
    MODEL_NAME = 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'
    MODEL_NAME_SHORT = MODEL_NAME.split('/')[-1]    # Used for saving results
    MAX_SEQ_LENGTH = 2048                               # Max. input length  
    BATCH_SIZE_PER_GPU = 2                              # Batch size PER GPU
    DTYPE = 'auto'                                      # 'None' for auto-detection (on unsloth)
    LOAD_IN_4_BIT = True                                # Reduces memory usage
    FT_SAVE_DIR = 'ft_outputs'                          # Directory to save the fine-tuned model
    FT_TOKENISER_DIR = 'ft_tokenisers'                  # Directory to save the fine-tuned tokeniser

    # Dataset settings
    DATASET_NAME = 'openai/gsm8k'
    SUBSET_NAME = 'main'
    SPLIT_NAME = 'train'

    #################################
    # DO NOT MODIFY BELOW THIS LINE #
    #################################

    # 1. Initialise the model and tokeniser
    login(token=HF_TOKEN)
    loaded_model, loaded_tokeniser = init(
        model_name = MODEL_NAME, 
        dtype = DTYPE, 
        load_in_4_bit = LOAD_IN_4_BIT
    )
    check_vram_usage()

    # 2. Load and format the dataset
    dataset = format_dataset(
        dataset_name = DATASET_NAME, 
        subset_name = SUBSET_NAME, 
        split_name = SPLIT_NAME, 
        tokeniser = loaded_tokeniser
    )
    check_vram_usage()

    # 3. Fine-tune the model
    fine_tune_model(
        model = loaded_model,
        tokeniser = loaded_tokeniser,
        dataset = dataset,
        max_seq_length = MAX_SEQ_LENGTH,
        batch_size_per_gpu = BATCH_SIZE_PER_GPU,
        model_name = MODEL_NAME_SHORT,
        dataset_name = DATASET_NAME.split('/')[-1]
    )

    # 4. Clear VRAM
    empty_vram(model = loaded_model)
