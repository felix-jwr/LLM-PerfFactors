import gc
import re
import os
import json
import torch
import random
import pynvml
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Union


def check_vram_usage(plot: bool = False) -> list:
    """
    Check the current VRAM usage using pynvml.

    args:
        plot: bool, Whether to plot the VRAM usage.

    returns:
        vram_usage: list, VRAM usage for each GPU.
    """

    pynvml.nvmlInit()               

    # Get the VRAM usage for each GPU
    vram_usage = []
    device_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]

    for dh in device_handles:
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(dh)
        vram_usage.append((memory_info.used // 1024**2) / 1000)   # Convert to GB

    # Plot the VRAM usage (if requested)
    if plot:
        plt.figure(figsize=(5, 2))
        plt.ylim(0, np.max(vram_usage))
        plt.bar(range(0, pynvml.nvmlUnitGetDeviceCount), vram_usage, color='blue', label='Used')
        plt.show()

    print(f'VRAM Usage: {[str(i) + "GB" for i in vram_usage]}')
    pynvml.nvmlShutdown()

    return vram_usage


def empty_vram(model = None, trainer = None) -> None:
    """
    Empty the VRAM by deleting all variables and running garbage collection.

    args:
        model: (any), Model to delete.
        trainer: (any), Trainer to delete.

    returns:
        None
    """

    try:
        del model
        del trainer
    except:
        pass

    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()


def extract_answer(text: str, eos: str = None, truth: bool = False) -> str:
    """
    Extract the last numerical answer from a piece of text using regular expressions.

    args:
        text: str, Example to process.
        eos: str, End of string delimiter.
        truth: bool, Whether the text is a ground truth answer.

    returns:
        output: str, The extracted numerical output from the text.
    """

    def get_boxed(text, number_regex, symbols_regex):
        # Try looking for a \boxed{}
        boxed = re.findall(r'\\boxed\{([^}]*)\}', text)

        if boxed:
            answer = number_regex.findall(boxed[-1])
            if type(answer) == list: # If more than one number gets detected in the \boxed{}
                if answer:
                    answer = answer[-1]
                    output = symbols_regex.sub('', answer)
                    
                    return output
            elif answer:
                output = symbols_regex.sub('', answer)
   
        return ''

    # She regular on my expression
    number_regex = re.compile(r'[,\$£%g]?(-?\d+(?:,\d+)*(?:\.\d+)?)')
    symbols_regex = re.compile(r'[,\$£€¥%g]')

    # If it's the ground truth, just return the number
    if truth:
        answer = re.split(r'####', text)[-1].strip()
        output = symbols_regex.sub('', answer)
        
        return output
    
    # Otherwise, try \boxed{}
    boxed = get_boxed(text, number_regex, symbols_regex)
    if boxed: 
        return boxed

    # Failing that, just get the last number, splitting on eos first (if we have one)
    if eos:
        text = re.split(re.escape(eos), text)[0].strip()

    all_numbers = number_regex.findall(text)
    
    if all_numbers:
        answer = all_numbers[-1].strip()
        output = symbols_regex.sub('', answer)

        return output
    
    # Finally, if there aren't any numbers, just resturn empty string
    return ''


def save_results(model_name: str, dataset_name: str, n_shot: int, use_cot: bool, results: Union[list, dict]) -> None:
    """
    Save a results .json from a model evaluation.

    args:
        model_name: str, The model name. Should ideally be the short name (i.e. excluding 'unsloth/').
        dataset_name: str, The name of the dataset, 
        n_shot: int, The number of example questions used for the n-shot test.
        use_cot: bool, Whether the 'Let's think step by step.' prompt was used.
        results: list or dict, The results of model evaluation.

    returns:
        None
    """

    # Clear up any slashes in the model name to avoid making directories
    model_name = model_name.replace('/', '_')
    dataset_name = dataset_name.replace('/', '_')
    dataset_name_short = dataset_name.split('_')[-1]

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f'../results/{dataset_name_short}/{model_name}', exist_ok=True)

    if use_cot:
        result_file = f'../results/{dataset_name_short}/{model_name}/{dataset_name}_{n_shot}-shot_cot_{timestamp}.json'
    else:
        result_file = f'../results/{dataset_name_short}/{model_name}/{dataset_name}_{n_shot}-shot_nocot_{timestamp}.json'

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Results saved to {result_file}')


def generate_n_shot_prompt(n_shot_data: dict, n: int, question: str, use_cot: bool, is_deepseek: bool = False,
                           is_mistral: bool = False) -> str:
    """
    Generate a prompt for the model with n example questions for an n-shot prompt.

    args:
        n_shot_data: dict, Training examples to use as n shots. Mustn't be from the test set.
        n: int, The number of example questions to include. Must be > 0.
        question: str, The actual prompt from the test set.
        seed: int, The random seed to use for reproducibility.
        use_cot: bool, Whether to use the 'Let's think step by step.' prompt.
        is_deepseek: bool, Whether the current model is DeepSeek (thus whether to use <think>\n at beginning of output)
        is_mistral: bool, Whether the current model is Mistral (Mistral doesn't accept system prompts)
        
    returns:
        string: str, The n-shot prompt for the model.
    """

    def question_prompt(string):
        return f'{string}'

    def answer_prompt(string, use_cot):
        if use_cot:
            return f'{string}'
        else:
            return f'The answer is {extract_answer(string, truth=True)}.'   # Only give the answer, not the working, for non-cot
        
    def generate_n_examples(n_shot_data, n, use_cot):
        n_shot_text = ''

        # Get random samples from the training set to use as n-shot examples
        for question_and_answer in random.sample(n_shot_data, n):
            n_shot_text += f'Question: {question_prompt(question_and_answer["question"])}'
            n_shot_text += f'\nAnswer: {answer_prompt(question_and_answer["answer"], use_cot = use_cot)}\n\n'

        return n_shot_text

    system_cot = 'You are a helpful AI assistant that will answer reasoning questions. You will reason step by step ' \
                'and you will always say at the end "$\\boxed{your answer}$". You must end your response with ' \
                '"\\boxed{your answer}" everytime!'
    system_nocot = 'You are a helpful AI assistant that will answer reasoning questions. You will only say ' \
                '"\\boxed{your answer}". You must end your response with "\\boxed{your answer}" everytime!'
    
    prompt = [] # Stores the prompt to give to the model
    system_prompt = f'System:\n{(system_cot if use_cot else system_nocot)}\n' if is_mistral else ''
    cot = 'Let\'s think step by step.' if use_cot else ''
    deepseek = '<think>\n' if is_deepseek else ''
    final_question = f'Question: {question} ' + cot + '\n' + 'Answer: ' + deepseek
    
    if not (is_mistral or is_deepseek):
        if use_cot:
            prompt.append(
                {'role': 'system', 'content': system_cot}
            )
        else:
            prompt.append(
                {'role': 'system', 'content': system_nocot}
            )

    # If number of shots > 0, generate the examples
    if n > 0:
        n_shots = system_prompt + '\n' + generate_n_examples(n_shot_data, n, use_cot)
        full_prompt = n_shots + final_question
        prompt.append(
            {'role': 'user', 'content': full_prompt}
        )
    # Otherwise, just append the actual question
    else:
        full_prompt = system_prompt + final_question
        prompt.append(
            {'role': 'user', 'content': full_prompt}
        )

    return prompt


def print_setup(parameters: dict = None) -> None:
    """
    Print the system information and parameters to the console.

    args:
        parameters: dict, The parameters to print.
    
    returns:
        None
    """

    # Print system information
    print('GPU:', torch.cuda.get_device_name())
    print('GPU VRAM:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
    print('CUDA Version:', torch.version.cuda)
    print('PyTorch Version:', torch.__version__)
    print('Python Version:', torch.__version__)
    print('Random Seed:', torch.initial_seed())
    print('Current Time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Print parameters
    if parameters:
        print('\nParameters:')
        for key, value in parameters.items():
            print(f'{key}: {value}')
    print()


def convert_prompt_format(raw_prompt: str) -> list:
    """
    Convert a prompt with special tokens into a clean turn-based conversation format.
    Works with any n-shot prompt format and handles the final question with no answer.
    
    Args:
        raw_prompt: String prompt potentially containing special tokens
        
    Returns:
        List of dictionaries with alternating user/assistant messages
    """

    clean_prompt = re.sub(r'<\|[^>]+\|>|begin_of_text|start_header_id|end_header_id|eot_id', '', raw_prompt)
    
    # Extract conversation turns using Q/A pattern
    qa_pairs = []
    pattern = r'(?:Q:|Question:)\s*(.*?)(?:(?:A:|Answer:)\s*(.*?)(?=(?:Q:|Question:)|$)|$)'
    matches = re.findall(pattern, clean_prompt, re.DOTALL)  # Include questions w/o answers for zero-shot
    
    # Convert matches to formatted conversation turns
    for question, answer in matches:
        question = question.strip()
        if question:  # If question exists
            qa_pairs.append({"role": "user", "content": question})
            
        answer = answer.strip()
        if answer:  # If answer exists
            qa_pairs.append({"role": "assistant", "content": answer})
    
    return qa_pairs


def make_json_safe(results: dict) -> dict:
    """
    Goes through data in a results dictionary and casts anything non-JSON-serialisable to a str, to prevent potential
    errors when saving results.

    args:
        results: dict, The results dict to make JSON.dump() safe.

    returns:
        safe_results: dict, The same results, but with non-JSON compatible types cast to str.
    """
    
    safe_results = {}

    for key, value in results.items():
        try:
            json.dumps(value)
            safe_results[key] = value

        except (TypeError, OverflowError) as e:
            print(f'WARN: Results at key {key} were not JSON-serialisable, casting to str. Make sure to check casted results.')
            
            # Recursively make nested dictionaries safe
            if isinstance(value, dict):
                safe_results[key] = make_json_safe(value) 

            # Handle lists by checking each element
            elif isinstance(value, (list, tuple)):
                
                safe_list = []
                for item in value:
                    if isinstance(item, dict):
                        safe_list.append(make_json_safe(item))
                    else:
                        try:
                            json.dumps(item)
                            safe_list.append(item)
                        except (TypeError, OverflowError):
                            safe_list.append(str(item))

                safe_results[key] = safe_list

            # Convert non-serialisable values to strings
            else:
                safe_results[key] = str(value)
    
    return safe_results
