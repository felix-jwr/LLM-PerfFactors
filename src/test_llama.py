import time
import torch
from tqdm import tqdm
from huggingface_hub import login
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from util import empty_vram, extract_answer, save_results, check_vram_usage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig


RANDOM_STATE = 42
HF_TOKEN = open('./hf_token.txt', 'r').read().strip()


def init(model_name: str, max_seq_length: int, dtype: str, load_in_4_bit: bool) -> tuple:
    """
    Initialise Model and Tokeniser.

    args:
        model_name: str, Name of the model to load from HF.
        max_seq_length: int, Maximum sequence length.
        bias: str, LoRA Bias (optimised for none).

    returns:
        model: AutoModelForCausalLM, The loaded HF model.
        tokeniser: AutoTokenizer, The loaded HF tokeniser.
    """

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit = load_in_4_bit,           # Activate 4-bit precision base model loading
    #     bnb_4bit_quant_type = 'nf4',            # Quantisation type (fp4 or nf4)
    #     bnb_4bit_compute_dtype = 'float16',     # Compute dtype for 4-bit base models
    #     bnb_4bit_use_double_quant = False,      # Activate nested quantisation for 4-bit base models (double quant)
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = dtype,
        # quantization_config = bnb_config,
        device_map = 'auto',
        trust_remote_code = True
    )

    tokeniser = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True
    )
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = 'left'    # Pipeline wants padding on the left

    return model, tokeniser


def format_dataset(dataset_name: str, subset_name: str, split_name: str, use_cot: bool, tokeniser) -> tuple:
    """
    Load a dataset from HF, and apply preprocessing (i.e. formatting prompts using the chat template appropriate for 
    the model used).

    args:
        dataset_name: str, The name of the dataset to load from HF.
        subset_name: str, The name of the subset of the dataset to load (e.g. 'main').
        split_name: str, The name of the split to load (e.g. 'train', 'test').
        use_cot: bool, Whether to use the 'Let's think step by step.' prompt.
        tokeniser: (any), The tokeniser to use for formatting the prompts.

    returns:
        dataset: dict, The loaded dataset, as is, without any additional processing.
        inputs: dict, The loaded dataset with prompts formatted in generic HF multi-turn conversation style.
    """

    # Load the dataset from HF
    dataset = load_dataset(dataset_name, subset_name, split=split_name)

    # Configure the tokenizer with the template
    def format_prompt(example):
        # For each entry in the dataset, apply CoT if required
        question = example['question'] # NOTE: the 'question' field may need to change dep. on dataset
        if use_cot: question = f'{question} Let\'s think step by step.'
        message = [{'role': 'user', 'content': f'{question}'}]

        # Apply the chat template
        formatted_prompt = tokeniser.apply_chat_template(
            message,
            tokenize=False, # We tokenise later
            add_generation_prompt=True, 
        )

        return formatted_prompt
    
    # Apply the formatting
    inputs = []
    for i in tqdm(range(len(dataset)), desc='Formatting prompts'):
        inputs.append( format_prompt(dataset[i]) )
    # inputs = Dataset.from_dict({'text': inputs})
    
    return dataset, inputs


def evaluate_model(model: dict, tokeniser: list, inputs: list, ground_truths: dict, chat_template: str = 'unsloth',
                   batch_size: int = 16) -> list:
    """
    Evaluate the model's accuracy given a set of decoded outputs and ground truths.

    args:
        model: AutoModelForCausalLM, The (loaded) model.
        tokeniser: (any), The (loaded) tokeniser.
        inputs: list, The inputs to the model.
        ground_truths: dict, The ground truths for the inputs.
        chat_template: str, The chat template to use for the model. Default is 'unsloth'.
        batch_size: int, The batch size to use for inference.

    returns:
        results: list, Prompt-response pairs, with the ground truth and correctness. Final element is overall accuracy.
    """

    # Configure model generator for inference
    tokeniser = get_chat_template(tokeniser, chat_template = chat_template)
    pipe = pipeline(
        'text-generation',
        model = model,
        tokenizer = tokeniser,
        max_new_tokens = MAX_SEQ_LENGTH,
        pad_token_id = tokeniser.eos_token_id,
        # Turns generation from O(n^3) to O(n^2): https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/2
        temperature = 1.5,
        # Use Temperature = 1.5, Min P = 0.1 because of this Tweet: https://x.com/menhguin/status/1826132708508213629
        min_p = 0.1,
        do_sample = True # Needed for Gemma 2
    )

    i = 0
    results = []
    total_correct = 0
    total_examples = len(inputs)
    start = time.time()
    print(f'Evaluating {total_examples} examples with batch size {batch_size}.')

    for output in tqdm(pipe(inputs, batch_size=batch_size), total=total_examples, desc='Evaluating'):
        correct = False

        # Get the model response
        output = output[0]['generated_text']
        ground_truth = ground_truths[i]['answer']

        # Extract numerical output
        extracted_output = extract_answer(output)
        extracted_ground_truth = extract_answer(ground_truth, truth=True)

        if extracted_output == extracted_ground_truth:
            correct = True
            total_correct += 1

        # Save prompt, reponse, ground truth, and correctness to be saved in .json file
        results.append({
            'model_prompt': inputs[i], 
            'model_response': output,
            'model_prediction': extracted_output,
            'ground_truth_text': ground_truth,
            'ground_truth_answer': extracted_ground_truth,
            'correct': correct
        })

        i += 1

    end = time.time()
    results.append({'accuracy': total_correct / total_examples})
    print(f'Accuracy: {total_correct} / {total_examples} = {total_correct / total_examples :.4f}')
    print(f'Finished in: {end - start:.2f}s ({(end - start) / total_examples:.2f}s per example)')

    return results


if __name__ == '__main__':
    #################################
    #            SETTINGS           #
    #################################
    
    # Loading the model
    MODEL_NAME = 'unsloth/phi-4-bnb-4bit'
    MODEL_NAME_SHORT = MODEL_NAME.split('/')[-1]        # Used for saving results
    CHAT_TEMPLATE_NAME = 'unsloth'                      # Chat template to use
    MAX_SEQ_LENGTH = 2048                               # Max. input length  
    BATCH_SIZE = 10                                     # Batch size for inference
    DTYPE = 'auto'                                      # 'None' for auto-detection (on unsloth)
    LOAD_IN_4_BIT = True                                # Reduces memory usage

    # Loading the dataset
    DATASET_NAME = 'openai/gsm8k'
    SUBSET_NAME = 'main'
    SPLIT_NAME = 'test'
    USE_COT = False
    N_SHOT = 0

    #################################
    # DO NOT MODIFY BELOW THIS LINE #
    #################################

    # 1. Initialise the model and tokeniser
    login(token=HF_TOKEN)
    loaded_model, loaded_tokeniser = init(
        model_name = MODEL_NAME, 
        max_seq_length = MAX_SEQ_LENGTH, 
        dtype = DTYPE, 
        load_in_4_bit = LOAD_IN_4_BIT
    )
    check_vram_usage()

    # 2. Load and format the dataset
    raw_dataset, model_prompts = format_dataset(
        dataset_name = DATASET_NAME, 
        subset_name = SUBSET_NAME, 
        split_name = SPLIT_NAME, 
        use_cot = USE_COT,
        tokeniser = loaded_tokeniser
    )
    check_vram_usage()

    # 3. Evaluate the model
    model_results = evaluate_model(
        model = loaded_model,
        tokeniser = loaded_tokeniser,
        chat_template = CHAT_TEMPLATE_NAME,
        inputs = model_prompts, 
        ground_truths = raw_dataset,
        batch_size = BATCH_SIZE
    )

    # 4. Save the results
    save_results(
        model_name = MODEL_NAME_SHORT, 
        dataset_name = DATASET_NAME, 
        n_shot = N_SHOT, 
        use_cot = USE_COT,
        results = model_results
    )

    # 5. Clear VRAM
    empty_vram(model = loaded_model, trainer = None)
