from test_mmlu import format_model_responses_gsm8k
import json
import re
from util import save_results

# Load the results from the JSON file
file_path = r"C:\Users\felix\VSCode Projects\MastersThesis\results\cot\Llama-3.2-3B-Instruct\gsm8k_cott_8-shot_cot_20250321_201318.json"

def convert_prompt_format(raw_prompt):
    """
    Convert a raw model prompt with special tokens to a turn-based format
    """
    formatted_prompts = []
    
    # Extract the main content by removing special tokens
    content = re.sub(r'<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>.*?<\|eot_id\|>', '', raw_prompt, flags=re.DOTALL)
    content = re.sub(r'<\|start_header_id\|>(user|assistant)<\|end_header_id\|>\n\n', '', content)
    content = re.sub(r'<\|eot_id\|>', '', content)
    
    # Split the content by "Q:" and "A:" to extract questions and answers
    qa_pairs = []
    questions = content.split("Q: ")
    
    for i in range(1, len(questions)):  # Skip the first empty split
        parts = questions[i].split("A: ", 1)
        if len(parts) == 2:
            question = parts[0].strip()
            
            # Handle the case where there's another question after this answer
            answer_parts = parts[1].split("Q: ", 1)
            answer = answer_parts[0].strip()
            
            qa_pairs.append({
                "role": "user", 
                "content": question
            })
            qa_pairs.append({
                "role": "assistant", 
                "content": answer
            })
    
    return qa_pairs



with open(file_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

for prompt_idx in range(len(results) - 1):
    results[prompt_idx]['model_prompt'] = convert_prompt_format(results[prompt_idx]['model_prompt'])

save_results(
    model_name="Llama-3.2-3B-Instruct",
    dataset_name="gsm8k_cot-Reformatted",
    n_shot=8,
    use_cot=True,
    results=results
)
