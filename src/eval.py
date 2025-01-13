import re
import json
import numpy as np

# TODO: This file may become deprecated with updates to test.py

################################################################################
# Load the json files
################################################################################

def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

# Load results
base_model_results = load_json("../results/20241121_144627_meta-llama_Llama-2-7b-chat-hf_outputs.json")
merged_model_results = load_json("../results/20241121_145010_.._llama-2-7b-ft-merged_outputs.json")

################################################################################
# Extract the answers
################################################################################

def extract_correct_answers(results):
    answers = []
    outputs = []

    for result in results:
        answers.append(result["Answer"])
        outputs.append(result["Output"])

    return answers, outputs

# Get answers and outputs
base_model_answers, base_model_outputs = extract_correct_answers(base_model_results)
merged_model_answers, merged_model_outputs = extract_correct_answers(merged_model_results)

################################################################################
# Evaluate the results
################################################################################

def extract_numerical_output(answer):
    # Extract the number from the answer
    match = re.search(r'\n#### (\d+)', answer)

    if match:
        return match.group(1)
    
    return None


def find_last_number(answer):
    # Find all numbers in the string
    matches = re.findall(r'\d+', answer)

    if matches:
        return matches[-1]  # Return the last number found
    
    return None


def calculate_mse(answers, outputs):
    answers = list(map(float, answers))
    outputs = list(map(float, outputs))
    mse = np.mean((np.array(answers) - np.array(outputs))**2)

    return mse


def evaluate(model, answers, outputs):
    correct = 0
    filtered_answers, filtered_outputs = [], []

    for i in range(len(answers)):
        answer = extract_numerical_output(answers[i])

        if model == "merged":
            output = extract_numerical_output(outputs[i])
        else:
            output = find_last_number(outputs[i])

        # Update correct total
        if answer == output:
            correct += 1

        # Filter out None values for MSE calculation
        if answer is not None and output is not None:
            filtered_answers.append(answer)
            filtered_outputs.append(output)

    accuracy = correct / len(answers)
    mse = calculate_mse(filtered_answers, filtered_outputs)

    return accuracy, mse

    
# Evaluate the models
base_model_acc, base_model_mse = evaluate("base", base_model_answers, base_model_outputs)
merged_model_acc, merged_model_mse = evaluate("merged", merged_model_answers, merged_model_outputs)

print(f"Base model accuracy: {round(base_model_acc*100, 2)}%")
print(f"Base model MSE: {round(base_model_mse, 2)}")
print(f"Merged model accuracy: {round(merged_model_acc*100, 2)}%")
print(f"Merged model MSE: {round(merged_model_mse, 2)}")