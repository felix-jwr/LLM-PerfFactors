import time
import torch
import random
import argparse
import numpy as np
import transformers
from tqdm import tqdm
from huggingface_hub import login
from datasets import load_dataset
# from unsloth.chat_templates import get_chat_template
from util import empty_vram, extract_answer, save_results, check_vram_usage, print_setup, generate_n_shot_prompt


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

    login(token=HF_TOKEN)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit = load_in_4_bit,                   # Activate 4-bit precision base model loading
        bnb_4bit_use_double_quant = True,               # Activate nested quant for 4-bit base models (double quant)
        bnb_4bit_quant_type = 'nf4',                    # Quantisation type (fp4 or nf4)
        bnb_4bit_compute_dtype = dtype,                 # Compute dtype for 4-bit base models
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = 'auto',
        quantization_config = bnb_config,
        token = HF_TOKEN,
    )

    tokeniser = transformers.AutoTokenizer.from_pretrained(
        model_name, 
        token=HF_TOKEN,
    )

    mistral_chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST]' + message['content'] + '[/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    tokeniser.chat_template = mistral_chat_template

    return model, tokeniser


def format_dataset(template_name: str, dataset_name: str, subset_name: str, n_shot: int, use_cot: bool, random_state: int, 
                   tokeniser) -> tuple:
    """
    Load a dataset from HF, and apply preprocessing (i.e. formatting prompts using the chat template appropriate for 
    the model used).

    args:
        template_name: str, The name of the chat template for the model.
        dataset_name: str, The name of the dataset to load from HF.
        subset_name: str, The name of the subset of the dataset to load (e.g. 'main').
        split_name: str, The name of the split to load (e.g. 'train', 'test').
        use_cot: bool, Whether to use the 'Let's think step by step.' prompt.
        random_state: int, Fix the random state (to ensure reproducability).
        tokeniser: (any), The tokeniser to use for formatting the prompts.

    returns:
        test_data: dict, The loaded dataset, as is, without any additional processing.
        inputs: dict, The loaded dataset with prompts formatted in generic HF multi-turn conversation style.
    """

    # Load the dataset from HF
    all_data = load_dataset(dataset_name, subset_name)
    test_data = all_data["test"].to_list()
    train_data = all_data["train"].to_list()

    # Check if the model is one which requires special handling of the input prompt
    is_deepseek = ('deepseek' in MODEL_NAME_SHORT.lower())
    is_mistral = ('mistral' or 'phi' in MODEL_NAME_SHORT.lower())

    # Format training data so they can be randomly sampled for n-shot prompts
    inputs = []
    for i in tqdm(range(len(test_data)), desc='Formatting prompts'):
        # The default gsm8k prompt from the CoT paper
        # https://arxiv.org/pdf/2201.11903.pdf page 35.
        prompt = generate_n_shot_prompt(
            n_shot_data = train_data, 
            n = n_shot, 
            question = test_data[i]['question'],
            use_cot = use_cot,
            is_deepseek = is_deepseek,
            is_mistral = is_mistral
        )
        
        # Don't need to tokenise here as the pipeline does it
        # prompt = tokeniser.apply_chat_template(
        #     prompt,
        #     return_tensors = 'pt',
        #     return_dict = True,
        #     padding = True).to('cuda')
        
        inputs.append( prompt )

    return test_data, inputs

def evaluate_model(model: dict, tokeniser: list, inputs: list, ground_truths: dict, batch_size: int = 1) -> list:
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
    # tokeniser = get_chat_template(tokeniser, chat_template = chat_template)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = 'left'    # NOTE: Pipeline wants padding on the left (?)
    pipe = transformers.pipeline(
        'text-generation',
        model = model,
        tokenizer = tokeniser,
        max_new_tokens = MAX_SEQ_LENGTH,
        pad_token_id = tokeniser.eos_token_id,
        # model_kwargs = {"torch_dtype": torch.bfloat16},
        # Turns generation from O(n^3) to O(n^2): https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/2
        use_cache = True,
        # temperature = 0.6, # TODO: ONLY ON WITH DEEPSEEK, DEEPSEEK RECOMMENDS TEMP = 0.6
        # Recommends Temp 1.5, Min_P 0.1: https://x.com/menhguin/status/1826132708508213629
        # temperature = 1.5,
        # min_p = 0.1,
    )

    start = time.time()
    results = []
    total = num_correct = 0
    total_examples = len(inputs)
    print(f'Evaluating {total_examples} examples with batch size {batch_size}.')

    for output in tqdm(pipe(inputs, batch_size = batch_size), total = total_examples, desc='Evaluating'):
        is_correct = False

        # Get the model response
        # [0] get dict, ['generated_text'] for output, [-1] for response to prompt, ['content'] for the actual text
        response = output[0]['generated_text'][-1]['content']
        ground_truth = ground_truths[total]['answer']

        # Extract numerical output
        extracted_output = extract_answer(response)
        extracted_ground_truth = extract_answer(ground_truth, truth=True)

        if extracted_output == extracted_ground_truth:
            is_correct = True
            num_correct += 1

        # Save prompt, reponse, ground truth, and correctness to be saved in .json file
        results.append({
            'model_prompt': inputs[total], 
            'model_response': output,
            'model_prediction': extracted_output,
            'ground_truth_text': ground_truth,
            'ground_truth_answer': extracted_ground_truth,
            'correct': is_correct
        })
        total += 1

    end = time.time()
    results.append({'accuracy': num_correct / total_examples})
    print(f'Accuracy: {num_correct} / {total_examples} = {num_correct / total_examples :.4f}')
    print(f'Finished in: {end - start:.2f}s ({(end - start) / total_examples:.2f}s per example)')

    return results


if __name__ == '__main__':
    #################################
    #     LOAD SETTINGS FROM CLI    #
    #################################

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run multi GPU inference test with LLM model')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='unsloth/Llama-3.1-8B-Instruct-bnb-4bit',
                        help='Name of the model to load from HF')
    parser.add_argument('--chat_template', type=str, default='unsloth',
                        help='Chat template to use')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Whether to load model in 4-bit precision')
    parser.add_argument('--no_4bit', action='store_false', dest='load_in_4bit',
                        help='Disable 4-bit quantization')
    
    # Dataset parameters
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Fix random seed to ensure reproducibility')
    parser.add_argument('--dataset', type=str, default='openai/gsm8k',
                        help='Dataset name to load from HF')
    parser.add_argument('--subset', type=str, default='main',
                        help='Dataset subset name')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split name')
    parser.add_argument('--use_cot', action='store_true', default=True,
                        help='Use chain-of-thought prompting')
    parser.add_argument('--no_cot', action='store_false', dest='use_cot',
                        help='Disable chain-of-thought prompting')
    parser.add_argument('--n_shot', type=int, default=0,
                        help='Number of examples for few-shot prompting')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size to use for inference')
    
    args = parser.parse_args()

    #################################
    #            SETTINGS           #
    #################################
    
    # Loading the model
    MODEL_NAME = args.model_name
    MODEL_NAME_SHORT = MODEL_NAME.split('/')[-1]        # Used for saving results
    CHAT_TEMPLATE_NAME = args.chat_template             # Chat template to use
    MAX_SEQ_LENGTH = args.max_seq_length                # Max. new output tokens 
    DTYPE = torch.bfloat16                              # 'None' for auto-detection
    LOAD_IN_4_BIT = args.load_in_4bit                   # Reduces memory usage

    # Loading the dataset
    RANDOM_STATE = args.random_seed
    DATASET_NAME = args.dataset
    SUBSET_NAME = args.subset
    SPLIT_NAME = args.split
    USE_COT = args.use_cot
    N_SHOT = args.n_shot
    BATCH_SIZE = args.batch_size

    #################################
    # DO NOT MODIFY BELOW THIS LINE #
    #################################

    # 0. Print settings
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    params = {
        'MODEL_NAME': MODEL_NAME,
        'MODEL_NAME_SHORT': MODEL_NAME_SHORT,
        'CHAT_TEMPLATE_NAME': CHAT_TEMPLATE_NAME,
        'MAX_SEQ_LENGTH': MAX_SEQ_LENGTH,
        'LOAD_IN_4_BIT': LOAD_IN_4_BIT,
        'DATASET_NAME': DATASET_NAME,
        'SUBSET_NAME': SUBSET_NAME,
        'SPLIT_NAME': SPLIT_NAME,
        'USE_COT': USE_COT,
        'N_SHOT': N_SHOT,
        'BATCH_SIZE': BATCH_SIZE
    }
    print_setup(parameters=params)

    # 1. Initialise the model and tokeniser
    loaded_model, loaded_tokeniser = init(
        model_name = MODEL_NAME, 
        max_seq_length = MAX_SEQ_LENGTH, 
        dtype = DTYPE, 
        load_in_4_bit = LOAD_IN_4_BIT
    )
    check_vram_usage()

    # 2. Load and format the dataset
    raw_dataset, model_prompts = format_dataset(
        template_name = CHAT_TEMPLATE_NAME, 
        dataset_name = DATASET_NAME, 
        subset_name = SUBSET_NAME, 
        n_shot = N_SHOT, 
        use_cot = USE_COT,
        random_state = RANDOM_STATE,
        tokeniser = loaded_tokeniser
    )

    # 3. Evaluate the model
    model_results = evaluate_model(
        model = loaded_model,
        tokeniser = loaded_tokeniser,
        inputs = model_prompts, 
        ground_truths = raw_dataset,
        # chat_template = 
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
