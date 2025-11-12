import os
from pathlib import Path

from datasets import load_dataset

def load_hf_token():
    """
    Pull HF_TOKEN from the repository-level .env so pushing to the Hub works
    without hardcoding secrets in source.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    hf_token = None
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("HF_TOKEN="):
                hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not hf_token:
        raise ValueError("Set HF_TOKEN in the repo .env file before running this script.")
    os.environ.setdefault("HF_TOKEN", hf_token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
    return hf_token


HF_TOKEN = load_hf_token()

RAW_DATASET_NAME = "ThomasTheMaker/Arc-Corpus"
DATA_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "hf-cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", str(DATA_CACHE_DIR))

raw_dataset_path = DATA_CACHE_DIR / "arc_raw"

if raw_dataset_path.exists():
    raw_dataset = load_dataset(
        raw_dataset_path.as_posix(),
        split="train",
    )
else:
    raw_dataset = load_dataset(
        RAW_DATASET_NAME,
        split="train",
        cache_dir=str(DATA_CACHE_DIR),
    )
    raw_dataset = raw_dataset.shuffle(seed=42)
    raw_dataset.save_to_disk(raw_dataset_path.as_posix())

def format_prompt(example):
    """
    Wrap raw dataset rows into the chat format expected by the tokenizer/model.
    """
    content = (example.get("text") or "").strip()
    if not content:
        content = "No content provided."
    conversation = (
        "<|user|>\n"
        "Share a helpful continuation for the following document.\n"
        "<|end|>\n"
        "<|bot|>\n"
        f"{content}\n"
        "<|end|>"
    )
    return {"text": conversation}

formatted_path = DATA_CACHE_DIR / "arc_formatted"

if formatted_path.exists():
    dataset = load_dataset(
        formatted_path.as_posix(),
        split="train",
    )
else:
    original_columns = raw_dataset.column_names
    dataset = raw_dataset.map(
        format_prompt,
        remove_columns=original_columns,
        desc="Formatting prompts",
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after formatting; please verify the source dataset.")
    dataset.save_to_disk(formatted_path.as_posix())

print(dataset[0]["text"])

def get_training_corpus():
    for row in raw_dataset:
        text = (row.get("text") or "").strip()
        if text:
            yield text

training_corpus = get_training_corpus()

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
)
from trl import SFTTrainer, SFTConfig

TINYLLAMA_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_VOCAB_SIZE = 128_256  # match tinyllama config

base_tokenizer = AutoTokenizer.from_pretrained(
    TINYLLAMA_TOKENIZER,
    use_fast=True,
)

# Use the default TinyLlama tokenizer (training disabled to save memory).
# tokenizer = base_tokenizer.train_new_from_iterator(
#     training_corpus,
#     vocab_size=TARGET_VOCAB_SIZE,
# )
tokenizer = base_tokenizer

if not isinstance(tokenizer, PreTrainedTokenizerFast):
    raise ValueError("Expected a fast tokenizer to attach chat template metadata.")

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "additional_special_tokens": ["<|user|>", "<|bot|>", "<|end|>"],
}
tokenizer.add_special_tokens(special_tokens)

tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")

chat_template = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|bot|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
    "{{ eos_token }}"
)
tokenizer.chat_template = chat_template

if tokenizer.pad_token_id is None:
    raise ValueError("Tokenizer is missing a pad token ID after special token setup.")

effective_vocab_size = len(tokenizer)

print(
    tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Why is the sky blue?"},
            {"role": "assistant", "content": "Due to Rayleigh scattering."},
        ],
        tokenize=False,
    )
)

# Configure ~800M parameter TinyLlama variant (embedding + transformer blocks).
config = LlamaConfig(
    vocab_size=effective_vocab_size,
    hidden_size=1536,
    intermediate_size=4096,
    num_hidden_layers=22,
    num_attention_heads=12,
    num_key_value_heads=3,
    max_position_embeddings=4096,
    rms_norm_eps=1.0e-6,
    initializer_range=0.02,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    tie_word_embeddings=False,
)

model = LlamaForCausalLM(config)

sft_config = SFTConfig(
    output_dir="output",
    num_train_epochs=1,
    max_steps=50_000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="adamw_torch",
    bf16=True,
    logging_steps=100,
    save_strategy="steps",
    save_steps=5_000,
    dataset_text_field="text",
    max_length=4_096,
    push_to_hub=True,
    hub_model_id="ThomasTheMaker/Arc",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    processing_class=tokenizer,
    train_dataset=dataset,
)

trainer.train()

tokenizer.save_pretrained("tokenizers/tinyllama")

trainer.push_to_hub(
    commit_message="Initial TinyLlama SFT run",
    token=HF_TOKEN,
)
