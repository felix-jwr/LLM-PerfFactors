import os
import datetime
from util import empty_vram, check_vram_usage, evaluate_perplexity
from merge import merge_weights_with_model


####################################################################################################
# Load Merged Model
####################################################################################################

# Define model parameters
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
device_map = "auto"
merged_model_dir = "../ft-model-merged"

# Merge weights with base model (if not already done)
if not os.path.exists('data'):
    merged_model_dir = merge_weights_with_model(model_name=base_model_name , device_map=device_map, output_dir=merged_model_dir)
    print(f"Merged model saved to {merged_model_dir}")
else:
    # Load the merged model
    model = AutoModelForCausalLM.from_pretrained(merged_model_dir, torch_dtype=torch.float16, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir)
    print(f"Merged model already exists at {merged_model_dir}")

####################################################################################################
# Load Dataset
####################################################################################################

# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, "main")
test_data = dataset["test"]

# Pre-process dataset
# Combine question and answer into a singe text field, allows for `group_by_length` batching and packing (efficiency)
def combine_text(examples):
    return {"text": examples["question"] + " " + examples["answer"]}

# Apply pre-processing
# Note this MAY (insufficicent testing) reduce model performance, so need to weigh efficiency vs. performance
test_data = test_data.map(combine_text)

# Check VRAM usage
check_vram_usage(plot=False)

####################################################################################################
# Evaluate Model
####################################################################################################

# Evaluate the model on the test dataset
perplexity, total_loss = evaluate_perplexity(model, tokenizer, test_data, max_length=300)
print(f"Perplexity: {perplexity}, Total Loss: {total_loss}")

# Write to file
now = datetime.now()
filename = f"../results/{now.strftime('%Y%m%d%H%M%S')}-results.txt"

with open(filename, "w") as f:
    f.write(f"Perplexity: {perplexity}, Total Loss: {total_loss}")
print(f"Results written to {now}")

# Empty VRAM
empty_vram(model=model, trainer=None)