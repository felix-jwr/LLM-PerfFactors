import torch
from datasets import load_dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from util import empty_vram, extract_answer, save_results


RANDOM_STATE = 42
HF_TOKEN = open('./hf_token.txt', 'r').read().strip()


def init_unsloth(model_name: str, max_seq_length: int, dtype: str, load_in_4_bit: bool) -> tuple:
    """
    Initialise Model and Tokeniser using Unsloth.

    args:
        model_name: str, Name of the model to load from HF
        max_seq_length: int, Maximum sequence length
        r: int, LoRA Rank
        lora_dropout: float, LoRA Dropout (optimised for 0)
        bias: str, LoRA Bias (optimised for none)

    returns:
        model: FastLanguageModel, The loaded HF model
        tokeniser: FastTokeniser, The loaded HF tokeniser
    """

    model, tokeniser = FastLanguageModel.for_inference(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4_bit = load_in_4_bit,
        token = HF_TOKEN    # TODO: Set this up with Docker secrets
    )

    # Add LoRA Adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_dropout = 0,
        bias = 'none',
        use_gradient_checkpointing = 'unsloth',
        random_state = RANDOM_STATE,
        use_rslora = False,
        loftq_confid = None
    )

    return model, tokeniser


def format_dataset(model_name: str, dataset_name: str, subset_name: str, split_name: str, use_cot: bool) -> tuple:
    """
    Load a dataset from HF, and apply preprocessing (i.e. formatting prompts using the chat template appropriate for 
    the model used).

    args:
        model_name: str, The name of the model being used, to get corresponding Unsloth chat template.
        dataset_name: str, The name of the dataset to load from HF.
        subset_name: str, The name of the subset of the dataset to load (e.g. 'main').
        split_name: str, The name of the split to load (e.g. 'train', 'test').

    returns:
        dataset: dict, The loaded dataset, as is, without any additional processing.
        inputs: dict, The loaded dataset with prompts formatted in generic HF multi-turn conversation style. # TODO: Probably not a dict
    """

    # Load the dataset from HF
    login(token = HF_TOKEN)
    dataset = load_dataset(dataset_name, subset_name, split=split_name)

    # Get the chat template for the model
    tokeniser = get_chat_template(
        tokeniser,
        chat_template = model_name
    )

    def format_prompts(examples: dict) -> dict:
        # This setup uses GSM8K dataset, because of the ['question'] key
        convos = []

        for i in range(len(examples)):
            if use_cot:
                convos.append( {'role': 'user', 'content': f'{examples[i]['question']} Let\'s think step by step.'} )
            else:
                convos.append( {'role': 'user', 'content': examples[i]['question']} )
        
        inputs = tokeniser.apply_chat_template(
            convos,
            tokenize = True,
            add_generation_prompt = True, # Needed for generation
            return_tensors = 'pt'
        ).to('cuda')

        return inputs
    return format_prompts(dataset)


def run_inference(model: FastLanguageModel, tokeniser: FastTokenizer, inputs: dict) -> list:
    """
    Run inference on the model, generating responses to the given inputs.

    args:
        model: FastLanguageModel, The (loaded) model.
        tokeniser: FastTokenizer, The (loaded) tokeniser.
        inputs: dict, The inputs to the model. # TODO: Probably not a dict
        ground_truths: dict, The ground truths for the inputs.

    returns:
        decoded_outputs: list, The decoded outputs from the model.
    """
    results = {}

    # Generate results
    outputs = model.generate(
        input_ids = inputs,
        tokeniser = tokeniser,
        max_new_tokens = 2048,
        pad_token_id = tokeniser.eos_token_id,
        # Turns generation from O(n^3) to O(n^2): https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/2
        use_cache = True, 
        # Use Temperature = 1.5, Min P = 0.1 because of this Tweet: https://x.com/menhguin/status/1826132708508213629
        temperature = 1.5, 
        min_p = 0.1
    )
    decoded_outputs = tokeniser.batch_decode(outputs)

    return decoded_outputs


def evaluate_model(inputs: dict, decoded_outputs: list, ground_truths: list) -> list:
    """
    Evaluate the model's accuracy given a set of decoded outputs and ground truths.

    args:
        inputs: dict, The prompts given to the model. # TODO: Probably not a dict
        decoded_outputs: list, The decoded outputs from the model.
        ground_truths: list, The ground truths for the inputs.

    returns:
        results: list, Prompt-response pairs, with the ground truth and correctness. Final element is overall accuracy.
    """

    # Construct list of dictionaries, containing prompt-response, ground truth, and correctness
    results = []

    for i in range(len(decoded_outputs)):
        results.append({
            'model_response': decoded_outputs[i],
            'model_prediction': extract_answer(decoded_outputs[i]),
            'ground_truth_text': ground_truths[i],
            'ground_truth_answer': extract_answer(ground_truths[i]),
            'correct': (extract_answer(decoded_outputs[i]) == extract_answer(ground_truths[i]))
        })

    # Get overall accuracy
    correct, total = 0
    for result in results:
        if result['correct']:
            correct += 1
        total += 1

    results.append({'accuracy': correct / total})
    print(f'Accuracy: {correct} / {total} = {correct / total :.4f}')

    return results


if __name__ == '__main__':
    #################################
    #            SETTINGS           #
    #################################
    
    # Loading the model
    MODEL_NAME = 'unsloth/Llama-3.2-1B-bnb-4bit'    # Model to load
    MODEL_NAME_SHORT = MODEL_NAME.split('/')[-1]    # Used for saving results
    MAX_SEQ_LENGTH = 2048                           # Max. input length  
    DTYPE = None                                    # 'None' for auto-detection
    LOAD_IN_4_BIT = True                            # Reduces memory usage

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

    # 2. Load and format the dataset
    raw_dataset, model_prompts = format_dataset(
        model_name = MODEL_NAME, 
        dataset_name = DATASET_NAME, 
        subset_name = SUBSET_NAME, 
        split_name = SPLIT_NAME, 
        use_cot = USE_COT
    )

    # 3. Run inference on the model
    model_outputs = run_inference(
        model = loaded_model, 
        tokeniser = loaded_tokeniser, 
        inputs = model_prompts
    )

    # 4. Evaluate the model
    model_results = evaluate_model(
        inputs = model_prompts, 
        decoded_outputs = model_outputs, 
        ground_truths = raw_dataset
    )

    # 5. Save the results
    save_results(
        model_name = MODEL_NAME_SHORT, 
        dataset_name = DATASET_NAME, 
        n_shot = N_SHOT, 
        results = model_results
    )

    # 6. Clear VRAM
    empty_vram(model = loaded_model, trainer = None)
