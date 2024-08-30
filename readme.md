For Dataset Creation
```
conda create -n dataset python=3.9 -y
conda activate dataset
pip install promptsource datasets
python dataset_creation.py
```
For MagMax

torch-2.5.0 transformers-4.40.0 evaluate-0.4.2

Sequential Finetuning with Merging and evaluation
```
conda create -n magmax python=3.12 -y
conda activate magmax
mkdir exp
mkdir exp/seq-fine
mkdir exp/mm
mkdir exp/mm-ot #for maximum fisher
bash run.sh
python extract_results.py
```