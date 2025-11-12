# Goal: create a dataset (parquet) called arc-corpus
# The dataset has 1 column called `text`, made of the 'content' column of these 2 dataset ThomasTheMaker/arc-stack-cpp-v2 & ThomasTheMaker/arc-stack-c-v2
# The 'text' column also has data from the 'text' column of ThomasTheMaker/arc-dclm and ThomasTheMaker/arc-fineweb
# Interweave and mix the data from the 4 datasets together, making sure to shuffle the data randomly, example below

'''
from datasets import interleave_datasets

mixed = interleave_datasets(
    [c_dataset, cpp_dataset, dclm_dataset, fineweb_dataset],
    probabilities=[0.17, 0.3, 0.25, 0.28],
    seed=42
)

'''

from datasets import load_dataset, interleave_datasets

# Load the datasets from HuggingFace
print("Loading C dataset...")
c_dataset = load_dataset("ThomasTheMaker/arc-stack-c-v2", split="train")

print("Loading C++ dataset...")
cpp_dataset = load_dataset("ThomasTheMaker/arc-stack-cpp-v2", split="train")

print("Loading DCLM dataset...")
dclm_dataset = load_dataset("ThomasTheMaker/arc-dclm", split="train")

print("Loading FineWeb dataset...")
fineweb_dataset = load_dataset("ThomasTheMaker/arc-fineweb", split="train")

# Rename 'content' column to 'text' for C and C++ datasets
print("Renaming columns...")
c_dataset = c_dataset.rename_column("content", "text")
cpp_dataset = cpp_dataset.rename_column("content", "text")

# Keep only the 'text' column for all datasets
c_dataset = c_dataset.remove_columns([col for col in c_dataset.column_names if col != "text"])
cpp_dataset = cpp_dataset.remove_columns([col for col in cpp_dataset.column_names if col != "text"])
dclm_dataset = dclm_dataset.remove_columns([col for col in dclm_dataset.column_names if col != "text"])
fineweb_dataset = fineweb_dataset.remove_columns([col for col in fineweb_dataset.column_names if col != "text"])

# Interleave the datasets with specified probabilities
# Using 'all_exhausted' strategy to include all data from all datasets
# Smaller datasets (C) will be cycled to match larger ones
print("Interleaving datasets...")
mixed = interleave_datasets(
    [c_dataset, cpp_dataset, dclm_dataset, fineweb_dataset],
    probabilities=[0.17, 0.3, 0.25, 0.28],
    seed=42,
    stopping_strategy="all_exhausted"
)

# Save the mixed dataset as a parquet file
print("Saving to parquet...")
mixed.to_parquet("arc-corpus.parquet")

print("Done! arc-corpus.parquet created successfully.")

