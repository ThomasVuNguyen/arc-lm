from datasets import load_dataset
import json

# Stream the Arc-Corpus dataset from HuggingFace (doesn't download the whole dataset)
print("Streaming Arc-Corpus dataset...")
dataset = load_dataset("ThomasTheMaker/Arc-Corpus", split="train", streaming=True)

# Take the first 1000 rows
print("Collecting first 1000 rows...")
sample_data = []
for i, row in enumerate(dataset):
    if i >= 1000:
        break
    sample_data.append(row)

# Save as JSON
print("Saving to JSON...")
with open("arc-corpus-sample-1000.json", "w") as f:
    json.dump(sample_data, f, indent=2)

print(f"Done! arc-corpus-sample-1000.json created with {len(sample_data)} rows.")
