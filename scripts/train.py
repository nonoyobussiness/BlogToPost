import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# -----------------------------
# Configuration
# -----------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "dataset.jsonl"
OUTPUT_DIR = REPO_ROOT / "model" / "blog-to-post-lora-v2"

MAX_LENGTH = 256
EPOCHS = 3


# -----------------------------
# Load model + tokenizer
# -----------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)


# -----------------------------
# Configure LoRA
# -----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# -----------------------------
# Format training examples
# -----------------------------

def format_example(example):
    return f"""### Instruction:
Summarize the following article as a short, engaging social media post.

### Article:
{example['blog']}

### Post:
{example['post']}"""


# -----------------------------
# Load full dataset
# -----------------------------

pairs = []

with DATASET_PATH.open(encoding="utf-8") as f:
    for line in f:
        pairs.append(json.loads(line))

print(f"Loaded {len(pairs)} training examples.")

texts = [format_example(pair) for pair in pairs]


# -----------------------------
# Tokenize
# -----------------------------

print("Tokenizing dataset...")

tokenized = tokenizer(
    texts,
    truncation=True,
    max_length=MAX_LENGTH,
    padding="max_length",
)

ds = Dataset.from_dict(tokenized)

ds = ds.map(
    lambda x: {"labels": x["input_ids"]}
)


# -----------------------------
# Training configuration
# -----------------------------

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=1,

    save_strategy="epoch",

    report_to=[],

    dataloader_pin_memory=False,
)


# -----------------------------
# Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
)


# -----------------------------
# Train
# -----------------------------

print("\nStarting full LoRA training...\n")

trainer.train()


# -----------------------------
# Save adapter
# -----------------------------

print("\nSaving LoRA adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\nTraining complete!")
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
