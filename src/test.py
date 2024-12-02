import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from util import (
    empty_vram, 
    save_results,
    extract_answer,
    generate_n_shot_prompt
)
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from merge import merge_weights_with_model


def load_model_and_tokeniser(base_model_name, ft_model_name, ft_weights_dir, use_base=True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Activate 4-bit precision base model loading
        bnb_4bit_quant_type='nf4',  # Quantisation type (fp4 or nf4)
        bnb_4bit_compute_dtype='float16',   # Compute dtype for 4-bit base models
        bnb_4bit_use_double_quant=False, # Activate nested quantisation for 4-bit base models (double quant)
    )
    
    # Either load the base model or the fine-tuned model
    if use_base:
        tokeniser = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokeniser.pad_token = tokeniser.eos_token
        tokeniser.padding_side = 'right'

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            quantization_config=bnb_config,
        )

        model_name = base_model_name
    else:
        model, tokeniser, _ = merge_weights_with_model(base_model_name=base_model_name , ft_weights_dir=ft_weights_dir)
        model_name = ft_model_name
    
    return model, tokeniser, model_name


def run_inference(model, tokeniser, dataset, datasize, n_shot, n_shot_data):
    results = []

    for i in tqdm(range(datasize), desc='Evaluating'):
        current_example = dataset[i]

        # Get a response from the model
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokeniser,
            max_new_tokens=1500,  # Changed for Llama 3.1-8B-Instruct
            pad_token_id=tokeniser.eos_token_id,
        )
    
        messages = generate_n_shot_prompt(n_shot_data=n_shot_data, n=n_shot, question=current_example['question'])
        response = (generator(messages)[0])['generated_text'][-1]['content']

        # Extract ground truth and prediction
        model_prediction = extract_answer(response)
        ground_truth_answer = extract_answer(current_example['answer'])

        # Generate results
        correct = (model_prediction == ground_truth_answer) if model_prediction is not None else False
        results.append({
            'question': current_example['question'],
            'ground_truth_text': current_example['answer'],
            'model_response': response,
            'model_prediction': model_prediction,
            'ground_truth_answer': ground_truth_answer,
            'correct': correct
        })

    return results


def eval(model, tokeniser, dataset, datasize, n_shot, n_shot_data):
    # Run Few-Shot
    results = run_inference(model, tokeniser, dataset, datasize, n_shot, n_shot_data)

    # Get accuracy
    count = 0
    for result in results:
        if result['correct']:
            count += 1
            
    total = len(results)
    assert total == datasize
    print(f'Accuracy: {count} / {datasize} = {count / datasize :.4f}')
    results.append({'accuracy': count / datasize})

    return results

################################################################################
# Main
################################################################################

if __name__ == '__main__':
    # Set the random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)

    # Model details
    base_model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    ft_weights_dir = '../3.1-8B-gsm8k-ft-weights'
    ft_model_name = 'meta-llama/Llama-3.1-8B-Instruct'

    # Login to HF
    access_key = os.environ['API_TOKEN']
    login(token=access_key)
    
    # Load everything
    all_data = load_dataset('openai/gsm8k', 'main')
    dataset = all_data['test']
    datasize = len(dataset)
    model, tokeniser, model_name = load_model_and_tokeniser(base_model_name, ft_model_name, ft_weights_dir, use_base=False)

    # Set up prompting
    n_shot_data = all_data['train']
    n_shot_data = n_shot_data.to_pandas()
    n_shot_data = n_shot_data.to_dict(orient='records')

    # Evaluate model in 8-shot setting
    print(f'EVALUATING 8-SHOT {model_name.upper()}')
    few_shot_results = eval(model, tokeniser, dataset, datasize, n_shot=8, n_shot_data=n_shot_data)
    save_results(few_shot_results, model_name, dataset_name='gsm8k', n_shot=8)

    # TODO: Fix zero-shot prompting/answer extraction for llama 2 7b
    # Evaluate model in zero-shot setting
    print(f'EVALUATING 0-SHOT {model_name.upper()}')
    zero_shot_results = eval(model, tokeniser, dataset, datasize, n_shot=0, n_shot_data=n_shot_data)
    save_results(zero_shot_results, model_name, dataset_name='gsm8k', n_shot=0)

    empty_vram()