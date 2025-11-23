#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample text from a trained checkpoint.
Usage:
    python experiments/sample.py --checkpoint output/StarmindZero-100M/checkpoint-420000
    python experiments/sample.py --checkpoint output/StarmindZero-100M/checkpoint-420000 --prompt "def fibonacci"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(checkpoint_path, device=None, dtype=None):
    """Load model and tokenizer from checkpoint."""
    print(f"[*] Loading model from {checkpoint_path}...")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if dtype is None:
        if torch.cuda.is_available():
            use_bf16 = torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            dtype = torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    
    # Fix pad token if it's the same as eos token
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=dtype,
        device_map=device if device == "cpu" else "auto",
    )
    
    if device != "cpu" and not hasattr(model, "device"):
        model = model.to(device)
    
    model.eval()
    
    print(f"[OK] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"     Device: {device}, dtype: {dtype}")
    print(f"     Pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")
    print()
    
    return model, tokenizer


def sample(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.3,
    do_sample=True,
    num_samples=1,
):
    """Generate text samples from the model."""
    print(f"[*] Generating {num_samples} sample(s)...")
    print(f"    Prompt: {prompt!r}")
    print(f"    Max tokens: {max_new_tokens}, Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}")
    print()
    
    # Tokenize prompt with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            do_sample=do_sample,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )
    
    # Decode and print results
    prompt_length = input_ids.shape[1]
    for i, output_ids in enumerate(outputs):
        # Only decode the newly generated tokens (skip the prompt)
        generated_ids = output_ids[prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"{'='*70}")
        print(f"Sample {i+1}:")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Sample text from a trained checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., output/StarmindZero-100M/checkpoint-420000)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random, default: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold (default: 0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Repetition penalty (higher = less repetition, default: 1.3)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device=args.device)
    
    # Generate samples
    sample(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.greedy,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()

