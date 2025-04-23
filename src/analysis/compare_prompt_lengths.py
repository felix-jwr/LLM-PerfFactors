import os
import json
import numpy as np

# Path to results
results_path = r'\path\to\results'

# Store all prompts and responses in a single dict
all_messages = {}
model_list = os.listdir(results_path)

for curr_model in model_list:
    curr_model_results_path = os.path.join(results_path, curr_model)
    curr_model_results_files = os.listdir(curr_model_results_path)
    curr_model_results_files = [i for i in curr_model_results_files if '.json' in i]    # Limit files to .json
    
    curr_model_messages = {
        'zero_shot_nocot': [],
        'zero_shot_cot': [],
        'few_shot_nocot': [],
        'few_shot_cot': []
    }

    # Extract model prompts and respones
    for curr_results_file in curr_model_results_files:
        curr_results_path = os.path.join(curr_model_results_path, curr_results_file)

        # Get current prompt type
        if '0-shot' in curr_results_file:
            if 'nocot' in curr_results_file:
                prompt_type = 'zero_shot_nocot'
            else:
                prompt_type = 'zero_shot_cot'
        elif '8-shot' in curr_results_file:
            if 'nocot' in curr_results_file:
                prompt_type = 'few_shot_nocot'
            else:
                prompt_type = 'few_shot_cot'

        # Extract messages
        with open(curr_results_path, 'r', encoding='utf-8') as file:
            file_contents = json.load(file)

            for sample in file_contents[:-1]:   # Loop excluding accuracy measurement
                try:
                    model_prompt = sample['model_prompt']
                    model_response = sample['model_response']

                    if (type(model_response) == list):
                        model_response = model_response[0]['generated_text'][-1]['content']

                    # If the model repeats the prompt we need to remove it
                    if isinstance(model_prompt, str) and isinstance(model_response, str) and model_prompt in model_response:
                        model_response = model_response.replace(model_prompt, "", 1)
                        model_response = model_response.strip()

                    curr_model_messages[prompt_type].append(model_response)
                except:
                    pass

    all_messages[curr_model] = curr_model_messages

# Now, loop through all the messages and get the average length for each prompt type
all_message_lens = {
    'zero_shot_nocot': [],
    'zero_shot_cot': [],
    'few_shot_nocot': [],
    'few_shot_cot': []
}

for model in all_messages.keys():
    for prompt_type in all_messages[model].keys():
        for message in all_messages[model][prompt_type]:
            all_message_lens[prompt_type].append(len(message))

print('Average Response Length by Prompt Type:')
for prompt_type in all_message_lens.keys():
    average_len = np.array(all_message_lens[prompt_type]).mean()
    print(prompt_type, average_len)