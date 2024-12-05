from datasets import load_dataset
import json
import os

# Step 1: Load the dataset
dataset = load_dataset("nyu-mll/multi_nli")

# Step 2: Select the 'train' split
train_split = dataset['train']

# Step 3: Convert the dataset to a list of dictionaries with each dictionary representing a single sample
# This will give us row-oriented data, suitable for JSON serialization
train_data_list = [
    {key: value for key, value in zip(train_split.features, values)}
    for values in zip(*train_split.to_dict().values())
]

# Step 4: Save the data to a JSON file
output_dir = "/home/zshiap/NLP/ind_pro/data/"
output_file = "multi_nli_json/all_data.json"
output_path = os.path.join(output_dir, output_file)

# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(train_data_list, f, ensure_ascii=False, indent=4)

print(f"Data has been saved to {output_path}")