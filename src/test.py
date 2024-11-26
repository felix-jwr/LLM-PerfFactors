import os
import re
import json
import time
import torch
import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList
)
from util import (
    empty_vram, 
    check_vram_usage, 
    extract_ground_truth_gsm8k,
    extract_predicted_gsm8k,
    SpecificStringStoppingCriteria,
    generate_model_answer,
    process_model_answers
)
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from merge import merge_weights_with_model


FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A:"""

ZERO_SHOT_PROMPT = """Q: {question}
A:"""

# Set the random seed for reproducibility
random_seed=42
torch.manual_seed(random_seed)
use_base = True

# Login to HF
access_key = os.environ['API_TOKEN']
login(token = access_key)
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
ft_weights_dir = "../llama-2-7b-ft-weights"
ft_model_name = "meta-llama/Llama-2-7b-chat-hf-ft-gsm8k"

# Either load the base model or the fine-tuned model
if use_base:
    tokeniser = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model_name = base_model_name
else:
    model, tokeniser, _ = merge_weights_with_model(base_model_name=base_model_name , ft_weights_dir=ft_weights_dir)
    model_name = ft_model_name
 
# Load the dataset
print("\nLoading dataset...")
dataset = load_dataset("openai/gsm8k", "main")
dataset = dataset["test"]
datasize = len(dataset)
print('\nTest set size:', datasize)

# Define a stopping condition for generation
generation_util = [
    "Q:",
    "</s>",
    "<|im_end|>"
]

################################################################################
# Inference
################################################################################

def run_inference(model, tokeniser, prompt_template):
    results = []

    for i in tqdm(range(datasize), desc='Evaluating'):
        current_example = dataset[i]

        # Run the prompt through the model
        input_text = prompt_template.format(question=current_example['question'])
        stop_criteria = SpecificStringStoppingCriteria(tokeniser, generation_util, len(input_text))
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        # Generate answers
        model_answers = [generate_model_answer(
            model, tokeniser, input_text, stopping_criteria_list
        )]

        # Extract answers
        majority_answer, numeric_answers = process_model_answers(model_answers)

        # Generate results
        ground_truth_answer = extract_ground_truth_gsm8k(current_example['answer'])
        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        results.append({
            'question': current_example['question'],
            'gold_answer_text': current_example['answer'],
            'model_answers_text': [ma['text'] for ma in model_answers],
            'extracted_model_answers': numeric_answers,
            'extracted_gold_answer': ground_truth_answer,
            'majority_answer': majority_answer,
            'correct': correct
        })

    return results

################################################################################
# Evaluation
################################################################################

def eval(model, tokeniser, prompt_type="few_shot", prompt_template=FEW_SHOT_PROMPT):
    # Run Few-Shot
    results = run_inference(model, tokeniser, prompt_template)

    # Get accuracy
    count = 0
    for result in results:
        if result['correct']:
            count += 1
            
    total = len(results)
    print(f"Accuracy: {count} / {total} = {count / total :.4f}")
    results.append({'accuracy': count / total})

    # Save results
    os.makedirs(f"../results/{model_name}/{prompt_type}", exist_ok=True)
    result_file = f"../results/{model_name}/{prompt_type}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")

# Run Few-Shot Evaluation
# print("Evaluting FEW-SHOT")
# eval(model, tokeniser, "few_shot", FEW_SHOT_PROMPT)

# Run Zero-Shot Evaluation
print("Evaluting ZERO-SHOT")
eval(model, tokeniser, "zero_shot", ZERO_SHOT_PROMPT)