#!/usr/bin/env python3
"""
Token counter for ThomasTheMaker/Arc-Corpus dataset using TinyLlama tokenizer
Uses multiprocessing to leverage 8 CPU cores for parallel processing
"""

import multiprocessing as mp
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np


def count_tokens_batch(texts, tokenizer_name):
    """
    Count tokens for a batch of texts
    
    Args:
        texts: List of text strings
        tokenizer_name: Name of the tokenizer to use
    
    Returns:
        Total token count for this batch
    """
    # Load tokenizer in each worker process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    total_tokens = 0
    for text in texts:
        if text:  # Skip empty texts
            tokens = tokenizer.encode(text, add_special_tokens=True)
            total_tokens += len(tokens)
    
    return total_tokens


def main():
    # Configuration
    DATASET_NAME = "ThomasTheMaker/Arc-Corpus"
    TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    NUM_CORES = 8
    BATCH_SIZE = 1000  # Number of texts per batch
    
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    print(f"Dataset size: {len(dataset)} rows")
    print(f"Using tokenizer: {TOKENIZER_NAME}")
    print(f"Number of CPU cores: {NUM_CORES}")
    
    # Extract all texts
    texts = dataset['text']
    
    # Split texts into batches for parallel processing
    batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batches.append(texts[i:i + BATCH_SIZE])
    
    print(f"Split into {len(batches)} batches of ~{BATCH_SIZE} texts each")
    
    # Create partial function with tokenizer name
    count_func = partial(count_tokens_batch, tokenizer_name=TOKENIZER_NAME)
    
    # Use multiprocessing to count tokens in parallel
    print(f"\nCounting tokens using {NUM_CORES} cores...")
    with mp.Pool(processes=NUM_CORES) as pool:
        # Use imap for progress tracking
        token_counts = list(tqdm(
            pool.imap(count_func, batches),
            total=len(batches),
            desc="Processing batches"
        ))
    
    # Sum up all token counts
    total_tokens = sum(token_counts)
    
    # Calculate statistics
    avg_tokens_per_text = total_tokens / len(texts)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total texts: {len(texts):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per text: {avg_tokens_per_text:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
