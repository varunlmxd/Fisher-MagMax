import os
import json
import csv

# Define the directories and JSON filenames
directories = [
    "exp/seq-fine/t5-small-wic",
    "exp/seq-fine/t5-small-wic-cb",
    "exp/seq-fine/t5-small-wic-cb-rte",
    "exp/seq-fine/t5-small-wic-cb-rte-copa",
    "exp/seq-fine/t5-small-wic-cb-rte-copa-boolq",
    "exp/seq-fine/t5-small-wic-cb-rte-copa-boolq-wsc",
    "exp/seq-fine/t5-small-wic-cb-rte-copa-boolq-wsc-multirc",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.1",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.2",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.3",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.4",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.5",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.6",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.7",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.8",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.9",
    "exp/mm/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-1.0",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.1",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.2",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.3",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.4",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.5",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.6",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.7",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.8",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-0.9",
    "exp/mm-ot/magmax-wic-cb-rte-copa-boolq-wsc-multirc-fim-1.0",
    
]
languages = ["wic", "cb", "rte", "copa", "boolq", "wsc", "multirc"]
json_filename_template = "{}_predict_results.json"
output_csv = "results.csv"

# Define the keys to extract from JSON
keys_to_extract = ["predict_exact_match"]

# Initialize a list to hold rows for CSV
header1 = ["model"]

for language in languages:
    header1.extend([language] * len(keys_to_extract))

csv_data = [header1]

# Process each directory
for directory in directories:
    row = [directory]
    
    for language in languages:
        json_filename = json_filename_template.format(language)
        json_path = os.path.join(directory, json_filename)
        if os.path.exists(json_path):
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
                values = [data.get(key, None) for key in keys_to_extract]
                rounded_values = [round(value, 4) for value in values]
                row.extend(rounded_values)
        else:
            row.extend(["N/A"] * len(keys_to_extract))
    
    csv_data.append(row)

# Write the CSV data to the output file
with open(output_csv, "w", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)

print(f"CSV file saved as {output_csv}")

