import torch
import random
import lm_eval
import argparse
import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from lm_eval.utils import setup_logging
from transformers import BitsAndBytesConfig
from util import print_setup, check_vram_usage, save_results, extract_answer, convert_prompt_format, make_json_safe


setup_logging("DEBUG") # Set up logging
HF_TOKEN = open('./hf_token.txt', 'r').read().strip()

def evaluate(model_name: str, max_new_tokens: int = None, use_cot: bool = False, task_names: list = ['haerae'], 
            single_gpu: bool = True, batch_size: int = 1, random_state: int = 42, load_in_4bit: bool = True,
            n_shot: int = 0) -> dict:
    """
    Function to load an evaluate a model on a (list) of tasks by calling the simple_evaluate() function from
    the LM-Evaluation-Harness.
    Note that this requires LM-Evaluation-Harness be installed in this repo. See README.md for details.

    args:
        model_name: str, The namme of the model to load.
        max_new_tokens: int, The maximum number of new tokens to generate.
        use_cot: bool, Whether to use CoT prompting.
        task_names: list, List of tasks to evaluate on.
        single_gpu: bool, Whether to use a single (or multi) GPU.
        batch_size: int, Batch size.
        random_state: int, Fix the random state for reproduciblity.
        load_in_4bit: bool, Whether to load with 4-bit quantisation
        n_shot: int, Number of shots to include in prompt.

    returns:
        returns desc.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,                    # Activate 4-bit precision base model loading
        bnb_4bit_use_double_quant = True,               # Activate nested quant for 4-bit base models (double quant)
        bnb_4bit_quant_type = 'nf4',                    # Quantisation type (fp4 or nf4)
        bnb_4bit_compute_dtype = torch.bfloat16,        # Compute dtype for 4-bit base models
    )

    # /****************************************************************************************************************
    # *
    # * THE FOLLOWING CODE BLOCK MAKES USE OF THE LM EVALUATION HARNESS FOR ITS `simple_evaluate()` FUNCTION
    # *
    # *    Title: Language Model Evaluation Harness
    # *    Author: EleutherAI
    # *    Date: 05/03/2025
    # *    Code version: 0.4.8
    # *    Availability: https://github.com/EleutherAI/lm-evaluation-harness
    # *
    # ****************************************************************************************************************/

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
        'cache_requests': True
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
    formatted_results = make_json_safe(results)

    # Extract actual results
    if 'gsm8k' in task_names or 'gsm8k_cot' in task_names:
        print('Formatting GSM8k Results...')
        formatted_results = format_model_responses_gsm8k(results = results, use_cot = use_cot)
    
    return formatted_results


def format_model_responses_gsm8k(results: dict, use_cot: bool):
    """
    Gets the raw responses from the LM-Evaluation-Harness output.

    args:
        results: dict, The LM-Evaluation-Harness output.
        use_cot: bool, Whether CoT prompting was used.

    returns:
        formatted_responses: dict, The re-formatted results.
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
    parser.add_argument('--no_4bit', action='store_false', dest='load_in_4bit',
                        help='Whether to load model in 4-bit precision')
    parser.add_argument('--single_gpu', action='store_true', default=True,
                        help='Run evaluation on single GPU')
    parser.add_argument('--multi_gpu',  action='store_false', dest='single_gpu',
                        help='Run evaluation on multiple GPUs')
    
    # Dataset parameters
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Fix random seed to ensure reproducibility')
    parser.add_argument('--task_names', nargs='+', default=['haerae'],
                        help='Task names to run via LM Evaluation Harness (incl. Custom Tasks)')
    parser.add_argument('--use_cot', action='store_true', default=True,
                        help='Use chain-of-thought prompting')
    parser.add_argument('--no_cot', action='store_false', dest='use_cot',
                        help='Disable chain-of-thought prompting')
    parser.add_argument('--n_shot', type=int, default=0,
                        help='Number of examples for few-shot prompting')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size to use for inference')
    parser.add_argument('--results_dir', type=str, default='haerae', 
                        help='Directory to save results under.')
    
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
    results_dir = args.results_dir

    # Needed for saving results
    if type(TASK_NAMES) == str:
        dataset_name = TASK_NAMES
    elif type(TASK_NAMES) == list:
        dataset_name = TASK_NAMES[0]

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

    # 1. Evaluate the model
    results = evaluate(
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

    # 2. Save the results
    save_results(
        model_name = MODEL_NAME_SHORT, 
        dataset_name = results_dir,
        n_shot = N_SHOT, 
        use_cot = USE_COT,
        results = results
    )
