from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model1', type=str, required=True, help='Path to the first model')
parser.add_argument('--model2', type=str, required=True, help='Path to the second model')
parser.add_argument('--model3', type=str, required=True, help='Path to the third model')
parser.add_argument('--model4', type=str, required=True, help='Path to the fourth model')
parser.add_argument('--model5', type=str, required=True, help='Path to the fifth model')
parser.add_argument('--model6', type=str, required=True, help='Path to the sixth model')
parser.add_argument('--model7', type=str, required=True, help='Path to the seventh model')
parser.add_argument('--lamda', type=float, required=True, help='lamda value')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the results')

args = parser.parse_args()

# Load the base model and tokenizer
base_model_name = 'google-t5/t5-small'
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load fine-tuned models and their Fisher matrices
models = [args.model1, args.model2, args.model3, args.model4, args.model5, args.model6, args.model7]
finetuned_models = []
fisher_matrices = []

for model_path in models:
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="cuda")
    finetuned_models.append(finetuned_model)
    fisher_matrix_path = os.path.join(f"{model_path}-fim", "pytorch_model.bin")
    fim = torch.load(fisher_matrix_path)
    fisher_matrices.append(fim)

# Function to extract task-specific vectors
def extract_task_vector(base_model, finetuned_model):
    task_vector = {}
    for name, param in base_model.state_dict().items():
        task_vector[name] = finetuned_model.state_dict()[name] - param
    return task_vector

# Extract task-specific vectors for each model
task_vectors = [extract_task_vector(base_model, model) for model in finetuned_models]

# Merge the task vectors based on Fisher matrices
model_counts = [0] * len(finetuned_models)
merged_vector = {}

for key in task_vectors[0].keys():
    max_abs_vector = task_vectors[0][key]
    model_selected = torch.ones_like(task_vectors[0][key])
    
    # if key not present perform normal operation without fisher
    # encoder.embed_tokens.weight Not in Fisher
    # decoder.embed_tokens.weight Not in Fisher
    # lm_head.weight Not in Fisher
    if key not in fisher_matrices[0]:
        print(key, "Not in Fisher")
        for i in range(1, len(task_vectors)):
            task_vector1 = max_abs_vector
            task_vector2 = task_vectors[i][key]
            
            condition = torch.abs(task_vector2) > torch.abs(task_vector1)
            max_abs_vector = torch.where(condition, task_vector2, max_abs_vector) # select max abs vector
            model_selected = torch.where(condition, torch.tensor(i + 1, device=model_selected.device), model_selected)
    else:
        fisher_selected = fisher_matrices[0][key]
        for i in range(1, len(task_vectors)):
            task_vector1 = max_abs_vector
            task_vector2 = task_vectors[i][key]
            fisher_matrix1 = fisher_selected
            fisher_matrix2 = fisher_matrices[i][key]

            condition = torch.abs(fisher_matrix2) > torch.abs(fisher_matrix1)
            max_abs_vector = torch.where(condition, task_vector2, max_abs_vector) # select max abs vector
            fisher_selected = torch.where(condition, fisher_matrix2, fisher_selected) # select fisher points of max abs vector
            model_selected = torch.where(condition, torch.tensor(i + 1, device=model_selected.device), model_selected)
        

    for i in range(len(finetuned_models)):
        model_counts[i] += torch.sum(model_selected == (i + 1)).item()

    merged_vector[key] = max_abs_vector

# Calculate and log model contributions
total_count = sum(model_counts)
model_percentages = [(count / total_count) * 100 for count in model_counts]

os.makedirs(args.save_dir, exist_ok=True)
file_path = os.path.join(args.save_dir, "contribution.txt")
with open(file_path, "w") as f:
    for i in range(len(finetuned_models)):
        f.write(f"Model {i + 1} selected {model_counts[i]} times ({model_percentages[i]:.2f}%)\n")

# Merge task vectors into the base model
new_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, device_map="cuda")
new_model_state_dict = new_model.state_dict()

print("lamda", args.lamda)
for key in new_model_state_dict.keys():
    new_model_state_dict[key] += (args.lamda * merged_vector[key])

# Save the new model
new_model.load_state_dict(new_model_state_dict)
torch.save(new_model.state_dict(), os.path.join(args.save_dir, "pytorch_model.bin"))
new_model.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)

print("Merging done and Model saved")
