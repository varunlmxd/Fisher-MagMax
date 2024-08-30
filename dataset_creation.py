from promptsource.templates import DatasetTemplates
from datasets import load_dataset, DatasetDict, Dataset
import copy
# Load the SuperGLUE dataset tasks
superglue_tasks = ["wic", "rte", "cb","wsc", "boolq", "multirc", "copa"]

# Iterate over each SuperGLUE task
for task in superglue_tasks:
    print(task)
    if task == "wsc":
        dataset = load_dataset("super_glue","wsc.fixed")
        templates = DatasetTemplates("super_glue/wsc.fixed")    
    else:
        dataset = load_dataset("super_glue",task)
        templates = DatasetTemplates(f"super_glue/{task}")
    template = []
    for temp in templates.all_template_names:
        template.append(temp)
    print(template[0])
    template = templates[template[0]]
    input=[]
    target=[]
    idx=[]
    for i in range(len(dataset['train'])):
        prompt = template.apply(copy.deepcopy(dataset['train'][i]))
        input.append(prompt[0])
        target.append(prompt[1])
        idx.append(i)
    data_train = Dataset.from_dict({"input":input,"target":target,"idx":idx})

    input=[]
    target=[]
    idx=[]
    for i in range(len(dataset['validation'])):
        prompt = template.apply(copy.deepcopy(dataset['validation'][i]))
        input.append(prompt[0])
        target.append(prompt[1])
        idx.append(i)
    data_val = Dataset.from_dict({"input":input,"target":target,"idx":idx})

    final_dataset = DatasetDict({
        "train":data_train,
        "validation":data_val
    })
    final_dataset.save_to_disk(f"datasett/{task}")
    print(final_dataset)
    print(dataset['train'][0])
    print(final_dataset['train'][0])
    print(dataset['validation'][0])
    print(final_dataset['validation'][0])