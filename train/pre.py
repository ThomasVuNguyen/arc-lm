import os
import math
import time
import random
from itertools import islice

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable must be set")

RAW_DATASET_NAME = "ThomasTheMaker/Arc-Corpus"
TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_DATASET_ROWS = 9600_000

OUTPUT_DIR = "output_arc_lm_100m"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLOCK_SIZE = 4096
BATCH_SIZE = 24
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 1
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.01
GRAD_CLIP = 1.0
LOG_EVERY = 50
SAVE_EVERY = 5_000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

print("📦 Loading dataset stream...")
stream_ds = load_dataset(
    RAW_DATASET_NAME,
    split="train",
    streaming=True,
    token=HF_TOKEN,
)

def ensure_text(example):
    content = (example.get("text") or "").strip()
    if not content:
        content = "No content provided."
    return {"text": content}

print("🔡 Loading tokenizer:", TOKENIZER_NAME)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
}

to_add = {k: v for k, v in special_tokens.items() if getattr(tokenizer, k, None) is None}
if to_add:
    print("➕ Adding special tokens:", to_add)
    tokenizer.add_special_tokens(to_add)

pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

print(f"✅ Tokenizer vocab size: {len(tokenizer)}")
print(f"   pad_id={pad_id}, bos_id={bos_id}, eos_id={eos_id}")
print()

formatted_stream = stream_ds.map(ensure_text)

print("📊 Estimating dataset size...")
sample_size = min(1000, MAX_DATASET_ROWS)
sample_tokens = 0

temp_stream = stream_ds.map(ensure_text)
for i, ex in enumerate(islice(temp_stream, sample_size)):
    text = ex["text"]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    sample_tokens += len(ids) + 1

avg_tokens_per_doc = sample_tokens / sample_size
print(f"   Sampled {sample_size} documents, avg {avg_tokens_per_doc:.1f} tokens/doc")

num_docs = MAX_DATASET_ROWS
estimated_tokens = int(num_docs * avg_tokens_per_doc)
print(f"   Using first {num_docs:,} documents")
print(f"   Estimated total tokens: {estimated_tokens:,}")

TOKENS_PER_STEP = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS
TOTAL_STEPS = (estimated_tokens * NUM_EPOCHS) // TOKENS_PER_STEP
print(f"📊 Training for {TOTAL_STEPS:,} steps ({NUM_EPOCHS} epoch(s))")
print(f"   Tokens per step: {TOKENS_PER_STEP:,}")
print(f"   Total tokens: {estimated_tokens * NUM_EPOCHS:,}")
print()
    
print()

peek = list(islice(stream_ds.map(ensure_text), 1))
print("🔎 Sample:")
print((peek[0]["text"] if peek else "<empty>")[:500])
print()

formatted_stream = stream_ds.map(ensure_text)

config = LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    max_position_embeddings=BLOCK_SIZE,
    rms_norm_eps=1e-6,
    initializer_range=0.02,
    use_cache=False,
    pad_token_id=pad_id,
    bos_token_id=bos_id,
    eos_token_id=eos_id,
    tie_word_embeddings=False,
)

print("🧩 Building model...")
model = LlamaForCausalLM(config)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and (not use_bf16)

if use_bf16:
    dtype = torch.bfloat16
elif use_fp16:
    dtype = torch.float16
else:
    dtype = torch.float32

model = model.to(device, dtype=dtype)

print(
    f"✅ Model ready: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
    f"dtype={dtype}, device={device}"
)
print()

def token_block_stream(hf_stream, tokenizer, block_size, eos_id):
    buffer = []
    
    for ex in hf_stream:
        text = ex["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eos_id)
        buffer.extend(ids)
        
        while len(buffer) >= block_size:
            block = buffer[:block_size]
            buffer = buffer[block_size:]
            yield torch.tensor(block, dtype=torch.long)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)

num_warmup_steps = int(TOTAL_STEPS * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=TOTAL_STEPS,
)

scaler = GradScaler(enabled=use_fp16)

print("🚀 Starting pretraining...")
print(
    f"   BLOCK_SIZE={BLOCK_SIZE}, BATCH_SIZE={BATCH_SIZE}, "
    f"GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}, TOTAL_STEPS={TOTAL_STEPS}"
)
print(
    f"   Effective tokens/step ≈ {BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS:,}"
)
print(f"   Learning rate: {LEARNING_RATE}, Warmup steps: {num_warmup_steps}")
print()

global_step = 0
micro_step = 0
running_loss = 0.0
start_time = time.time()
window_start_time = time.time()
window_start_step = 0

loss_history = []
lr_history = []
throughput_history = []
step_history = []

def multi_epoch_stream(base_stream, num_epochs, max_rows):
    for epoch in range(num_epochs):
        print(f"📚 Starting epoch {epoch + 1}/{num_epochs}")
        row_count = 0
        for item in base_stream:
            if row_count >= max_rows:
                break
            yield item
            row_count += 1
        print(f"   Processed {row_count:,} rows in epoch {epoch + 1}")

formatted_stream_base = stream_ds.map(ensure_text)
multi_epoch_data = multi_epoch_stream(formatted_stream_base, NUM_EPOCHS, MAX_DATASET_ROWS)
block_iter = token_block_stream(multi_epoch_data, tokenizer, BLOCK_SIZE, eos_id)

model.train()

pbar = tqdm(total=TOTAL_STEPS, desc="Training", unit="step")

autocast_ctx = autocast(enabled=(use_bf16 or use_fp16), dtype=torch.bfloat16 if use_bf16 else torch.float16)
with autocast_ctx:
    while global_step < TOTAL_STEPS:
        blocks = []
        for _ in range(BATCH_SIZE):
            try:
                block = next(block_iter)
                blocks.append(block)
            except StopIteration:
                print(f"\n✅ Dataset exhausted after {global_step} steps")
                break
        
        if len(blocks) < BATCH_SIZE:
            print(f"   Completed training with partial batch of {len(blocks)} blocks")
            break
        
        input_ids = torch.stack(blocks).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        labels = input_ids.clone()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / GRAD_ACCUM_STEPS
        
        if use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        running_loss += loss.item()
        micro_step += 1
        
        if micro_step % GRAD_ACCUM_STEPS == 0:
            if use_fp16:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            global_step += 1
            pbar.update(1)
            
            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                current_lr = scheduler.get_last_lr()[0]
                
                window_elapsed = time.time() - window_start_time
                window_steps = global_step - window_start_step
                tok_per_step = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS
                window_tps = (tok_per_step * window_steps) / window_elapsed if window_elapsed > 0 else 0
                
                total_elapsed = time.time() - start_time
                total_tps = (tok_per_step * global_step) / total_elapsed if total_elapsed > 0 else 0
                
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "tok/s": f"{int(window_tps):,}"
                })
                
                running_loss = 0.0
                window_start_time = time.time()
                window_start_step = global_step
            
            if global_step % SAVE_EVERY == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                print(f"\n💾 Saving checkpoint to {ckpt_dir}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                
                torch.save({
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
                }, os.path.join(ckpt_dir, "training_state.pt"))

pbar.close()

print("\n✅ Training complete!")
print("💾 Saving final model...")

final_dir = os.path.join(OUTPUT_DIR, "final-model")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

torch.save({
    'global_step': global_step,
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
}, os.path.join(final_dir, "training_state.pt"))

print("🎉 Done!")
