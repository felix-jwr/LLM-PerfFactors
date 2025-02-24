import time
import copy
import torch
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from util import empty_vram, extract_answer, save_results, check_vram_usage


RANDOM_STATE = 42
HF_TOKEN = open('./hf_token.txt', 'r').read().strip()


def init_unsloth(model_name: str, max_seq_length: int, dtype: str, load_in_4_bit: bool) -> tuple:
    """
    Initialise Model and Tokeniser using Unsloth.

    args:
        model_name: str, Name of the model to load from HF
        max_seq_length: int, Maximum sequence length

    returns:
        model: FastLanguageModel, The loaded HF model
        tokeniser: FastTokeniser, The loaded HF tokeniser
    """

    model, tokeniser = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4_bit,
        device_map = 'auto',
        token = HF_TOKEN
    )

    # # Add LoRA Adapter
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 16,
    #     lora_dropout = 0,
    #     bias = 'none',
    #     use_gradient_checkpointing = 'unsloth',
    #     random_state = RANDOM_STATE,
    #     use_rslora = False,
    #     loftq_config = None
    # )

    return model, tokeniser


def format_dataset(model_name: str, dataset_name: str, subset_name: str, split_name: str, use_cot: bool, tokeniser) -> tuple:
    """
    Load a dataset from HF, and apply preprocessing (i.e. formatting prompts using the chat template appropriate for 
    the model used).

    args:
        model_name: str, The name of the model being used, to get corresponding Unsloth chat template.
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
    login(token=HF_TOKEN)
    dataset = load_dataset(dataset_name, subset_name, split=split_name)

    # Prompt format
    prompt_format = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Response:
    {}
    """

    # Apply the formatting
    inputs = []
    EOS_TOKEN = tokeniser.eos_token

    for i in tqdm(range(len(dataset)), desc='Formatting prompts'):
        question = dataset[i]['question'] # NOTE: the 'question' field may need to change dep. on dataset
        
        if use_cot: 
            question = f'{question} Let\'s think step by step.'

        question = prompt_format.format(question, '')   # Leave response blank for generation
        inputs.append( question + EOS_TOKEN)
    
    return dataset, inputs


def run_inference(model: FastLanguageModel, tokeniser, input: torch.Tensor) -> tuple:
    """
    Run inference on the model, generating responses to the given inputs.

    args:
        model: FastLanguageModel, The (loaded) model.
        tokeniser: (any), The (loaded) tokeniser.
        input: tensor, The inputs to the model.

    returns:
        decoded_input: list, The decoded input from the model.
        decoded_output: list, The decoded output from the model.
    """

    output = model.generate(
        input_ids = input,
        tokenizer = tokeniser,
        max_new_tokens = MAX_SEQ_LENGTH,
        pad_token_id = tokeniser.eos_token_id,
        # Turns generation from O(n^3) to O(n^2): https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/2
        use_cache = True, 
        # Use Temperature = 1.5, Min P = 0.1 because of this Tweet: https://x.com/menhguin/status/1826132708508213629
        temperature = 1.5, 
        min_p = 0.1
    )

    decoded_input = tokeniser.batch_decode(input, skip_special_tokens=True)
    decoded_output = tokeniser.batch_decode(output, skip_special_tokens=True)

    return decoded_input, decoded_output


def evaluate_model(model: dict, tokeniser: list, inputs: list, ground_truths: dict) -> list:
    """
    Evaluate the model's accuracy given a set of decoded outputs and ground truths.

    args:
        model: FastLanguageModel, The (loaded) model.
        tokeniser: (any), The (loaded) tokeniser.
        inputs: list, The inputs to the model.
        ground_truths: dict, The ground truths for the inputs.

    returns:
        results: list, Prompt-response pairs, with the ground truth and correctness. Final element is overall accuracy.
    """

    results = []
    total_correct = 0
    total_examples = len(inputs)
    FastLanguageModel.for_inference(model)

    for i in tqdm(range(total_examples), desc='Processing Results'):
        correct = False

        # Get the model's response
        tokenised_inputs = tokeniser(inputs[i], return_tensors = 'pt', padding = True).to('cuda')
        output = model.generate(**tokenised_inputs, max_new_tokens=MAX_SEQ_LENGTH, use_cache=True)
        decoded_output = tokeniser.batch_decode(output, skip_special_tokens=True)
        ground_truth = ground_truths[i]['answer']

        # Extract numerical output
        extracted_output = extract_answer(decoded_output[0])
        extracted_ground_truth = extract_answer(ground_truth, truth=True)

        if extracted_output == extracted_ground_truth:
            correct = True
            total_correct += 1

        # Save prompt, reponse, ground truth, and correctness to be saved in .json file
        results.append({
            'model_prompt': inputs[i], 
            'model_response': decoded_output,
            'model_prediction': extracted_output,
            'ground_truth_text': ground_truth,
            'ground_truth_answer': extracted_ground_truth,
            'correct': correct
        })

    results.append({'accuracy': total_correct / total_examples})
    print(f'Accuracy: {total_correct} / {total_examples} = {total_correct / total_examples :.4f}')

    return results


if __name__ == '__main__':
    #################################
    #            SETTINGS           #
    #################################
    
    # Loading the model
    MODEL_NAME = 'unsloth/gemma-2-9b-it-bnb-4bit'
    MODEL_NAME_SHORT = MODEL_NAME.split('/')[-1]    # Used for saving results
    CHAT_TEMPLATE_NAME = 'gemma'                        # Chat template to use
    MAX_SEQ_LENGTH = 2048                               # Max. input length  
    DTYPE = None                                        # 'None' for auto-detection
    LOAD_IN_4_BIT = True                                # Reduces memory usage

    # Loading the dataset
    DATASET_NAME = 'openai/gsm8k'
    SUBSET_NAME = 'main'
    SPLIT_NAME = 'test'
    USE_COT = True
    N_SHOT = 0

    #################################
    # DO NOT MODIFY BELOW THIS LINE #
    #################################

    # 1. Initialise the model and tokeniser
    loaded_model, loaded_tokeniser = init_unsloth(
        model_name = MODEL_NAME, 
        max_seq_length = MAX_SEQ_LENGTH, 
        dtype = DTYPE, 
        load_in_4_bit = LOAD_IN_4_BIT
    )

    check_vram_usage()

    # 2. Load and format the dataset
    raw_dataset, model_prompts = format_dataset(
        model_name = CHAT_TEMPLATE_NAME, 
        dataset_name = DATASET_NAME, 
        subset_name = SUBSET_NAME, 
        split_name = SPLIT_NAME, 
        use_cot = USE_COT,
        tokeniser = loaded_tokeniser
    )

    # 3. Evaluate the model
    model_results = evaluate_model(
        model = loaded_model,
        tokeniser = loaded_tokeniser,
        inputs = model_prompts, 
        ground_truths = raw_dataset
    )

    # 4. Save the results
    save_results(
        model_name = MODEL_NAME_SHORT, 
        dataset_name = DATASET_NAME, 
        n_shot = N_SHOT, 
        results = model_results
    )

    # 6. Clear VRAM
    empty_vram(model = loaded_model, trainer = None)
