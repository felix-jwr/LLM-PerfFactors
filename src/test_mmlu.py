import torch
import random
import lm_eval
import argparse
import numpy as np
from huggingface_hub import login
from datasets import load_dataset
# from lm_eval.api.model import LM
from lm_eval.utils import setup_logging
from transformers import BitsAndBytesConfig
from util import print_setup, check_vram_usage, save_results, extract_answer, convert_prompt_format, make_json_safe


setup_logging("DEBUG") # optional, but recommended; or you can set up logging yourself
HF_TOKEN = open('./hf_token.txt', 'r').read().strip()

# indexes all tasks from the `lm_eval/tasks` subdirectory.
# Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# to include a set of tasks in a separate directory.
# task_manager = lm_eval.tasks.TaskManager()

# Setting `task_manager` to the one above is optional and should generally be done
# if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# `simple_evaluate` will instantiate its own task_manager if it is set to None here.
def evaluate_mmlu(model_name: str, max_new_tokens: int = None, use_cot: bool = False, task_names: list = ['blimp'], 
                  single_gpu: bool = True, batch_size: int = 1, random_state: int = 42, load_in_4bit: bool = True,
                  n_shot: int = 0) -> dict:
    """
    Function desc.

    args:
        args desc.

    returns:
        returns desc.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,                    # Activate 4-bit precision base model loading
        bnb_4bit_use_double_quant = True,               # Activate nested quant for 4-bit base models (double quant)
        bnb_4bit_quant_type = 'nf4',                    # Quantisation type (fp4 or nf4)
        bnb_4bit_compute_dtype = torch.bfloat16,        # Compute dtype for 4-bit base models
    )

    # MUST have this when using custom tasks like haerae_cot
    task_manager = lm_eval.tasks.TaskManager(include_path='custom_tasks')

    # Set up model args based on whether we're using a single GPU
    model_args = {
        'pretrained': model_name, 
        'quantization_config': bnb_config,
        'trust_remote_code': True,
    }
    if not single_gpu:
        model_args['parallelize'] = True
    
    # Create base parameters that are common to both cases
    eval_params = {
        'model': 'hf',
        'model_args': model_args,
        'tasks': task_names,
        'batch_size': batch_size,
        'log_samples': True,
        'apply_chat_template': True,
        'random_seed': random_state,
        'numpy_random_seed': random_state,
        'torch_random_seed': random_state,
        'num_fewshot': n_shot,
        'task_manager': task_manager,
    }

    # If max_new_tokens is set
    if max_new_tokens is not None:
        eval_params['gen_kwargs'] = {'max_new_tokens': max_new_tokens}

    # Only specify device for single GPU setup
    if single_gpu:
        eval_params['device'] = 'cuda:0'
    
    # Evaluate with the appropriate parameters
    results = lm_eval.simple_evaluate(**eval_params)

    # Cast values which are not JSON serialisable to string
    results = make_json_safe(results)

    # Extract actual results
    # formatted_results = format_model_responses_gsm8k(results = results, use_cot = use_cot)
    formatted_results = results
    
    return formatted_results


def format_model_responses_gsm8k(results: dict, use_cot: bool):
    """
    get raw responses for custom accuracy calc cuz lm harness one sucks
    """

    ground_truths = load_dataset('openai/gsm8k', 'main', split='test')

    if use_cot:
        samples = results['samples']['gsm8k_cot']
    else:
        samples = results['samples']['gsm8k']
    
    correct = 0
    total_examples = len(ground_truths)
    formatted_responses = []
    for idx in range(total_examples):
            is_correct = False
            model_prompt = samples[idx]['arguments'][0][0]
            model_response = samples[idx]['resps'][0][0]
            model_prediction = extract_answer(model_response)
            ground_truth = ground_truths[idx]['answer']
            ground_truth_answer = extract_answer(ground_truth, truth=True)

            if model_prediction == ground_truth_answer:
                is_correct = True
                correct += 1

            # Convert to turn-based format
            if idx == 0: 
                print(f'Model prompt before splitting into q-a pairs: {model_prompt}')
            model_prompt = convert_prompt_format(model_prompt)
            if idx == 0:
                print(f'Model prompt after splitting into q-a pairs: {model_prompt}')

            formatted_responses.append({
                'model_prompt': model_prompt, 
                'model_response': model_response,
                'model_prediction': model_prediction,
                'ground_truth_text': ground_truth,
                'ground_truth_answer': ground_truth_answer,
                'correct': is_correct
            })

    formatted_responses.append({'accuracy': correct / total_examples})
    print(f'Accuracy: {correct} / {total_examples} = {correct / total_examples :.4f}')

    return formatted_responses


# TODO: TODO LIST
    # TODO: No custom filters defined
    # 2025-03-21:14:34:30 DEBUG    [tasks:539] File _evalita-mp_ner_adg.yaml in C:\Users\felix\VSCode Projects\MastersThesis\src\lm-evaluation-harness\lm_eval\tasks/evalita_llm could not be loaded
    # 2025-03-21:14:34:30 DEBUG    [tasks:539] File _evalita-mp_ner_fic.yaml in C:\Users\felix\VSCode Projects\MastersThesis\src\lm-evaluation-harness\lm_eval\tasks/evalita_llm could not be loaded
    # 2025-03-21:14:34:30 DEBUG    [tasks:539] File _evalita-mp_ner_wn.yaml in C:\Users\felix\VSCode Projects\MastersThesis\src\lm-evaluation-harness\lm_eval\tasks/evalita_llm could not be loaded
if __name__ == '__main__':
    #################################
    #     LOAD SETTINGS FROM CLI    #
    #################################

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run multi GPU inference test with LLM model')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Name of the model to load from HF')
    parser.add_argument('--chat_template', type=str, default='llama',
                        help='Chat template to use')
    parser.add_argument('--max_seq_length', type=int, default=None,
                        help='Maximum sequence length')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Whether to load model in 4-bit precision')
    parser.add_argument('--single_gpu', action='store_true', default=True,
                        help='Run evaluation on single GPU')
    parser.add_argument('--multi-gpu',  action='store_false', dest='single_gpu',
                        help='Run evaluation on multiple GPUs')
    
    # Dataset parameters
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Fix random seed to ensure reproducibility')
    parser.add_argument('--task_names', nargs='+', default=['blimp'],
                        help='Task names to run via LM Evaluation Harness (incl. Custom Tasks)')
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
    
    # Load parameters via argparse
    MODEL_NAME = args.model_name
    MODEL_NAME_SHORT = MODEL_NAME.split('/')[-1]        # Used for saving results
    CHAT_TEMPLATE_NAME = args.chat_template             # Chat template to use
    MAX_SEQ_LENGTH = args.max_seq_length                # Max. new output tokens
    LOAD_IN_4BIT = args.load_in_4bit
    SINGLE_GPU = args.single_gpu 
    RANDOM_STATE = args.random_seed
    TASK_NAMES = args.task_names
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
    login(token=HF_TOKEN)
    params = {
        'MODEL_NAME': MODEL_NAME,
        'MODEL_NAME_SHORT': MODEL_NAME_SHORT,
        'CHAT_TEMPLATE_NAME': CHAT_TEMPLATE_NAME,
        'MAX_SEQ_LENGTH': MAX_SEQ_LENGTH,
        'LOAD_IN_4BIT': LOAD_IN_4BIT,
        'SINGLE_GPU': SINGLE_GPU,
        'RANDOM_STATE': RANDOM_STATE,
        'TASK_NAMES': TASK_NAMES,
        'USE_COT': USE_COT,
        'N_SHOT': N_SHOT,
        'BATCH_SIZE': BATCH_SIZE
    }
    print_setup(parameters=params)

    # 1. Initialise the model and tokeniser
    # loaded_model, loaded_tokeniser = init(
    #     model_name = MODEL_NAME, 
    #     max_seq_length = MAX_SEQ_LENGTH, 
    #     dtype = DTYPE, 
    #     load_in_4_bit = LOAD_IN_4_BIT
    # )
    # check_vram_usage()

    # 2. Load and format the dataset
    # raw_dataset, model_prompts = format_dataset(
    #     template_name = CHAT_TEMPLATE_NAME, 
    #     dataset_name = DATASET_NAME, 
    #     subset_name = SUBSET_NAME, 
    #     n_shot = N_SHOT, 
    #     use_cot = USE_COT,
    #     random_state = RANDOM_STATE,
    #     tokeniser = loaded_tokeniser
    # )

    # 3. Evaluate the model
    results = evaluate_mmlu(
        model_name = MODEL_NAME,
        max_new_tokens = MAX_SEQ_LENGTH,
        use_cot = USE_COT,
        task_names = TASK_NAMES,
        single_gpu = SINGLE_GPU,
        batch_size = BATCH_SIZE,
        random_state = RANDOM_STATE,
        load_in_4bit = LOAD_IN_4BIT,
        n_shot = N_SHOT
    )

    # 4. Save the results
    save_results(
        model_name = MODEL_NAME_SHORT, 
        dataset_name = 'mmlu-haerae',
        n_shot = N_SHOT, 
        use_cot = USE_COT,
        results = results
    )

    # 5. Clear VRAM
    # empty_vram(model = loaded_model, trainer = None)
