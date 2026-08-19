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
# Model
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "dataset.jsonl"
SMOKE_OUTPUT_DIR = REPO_ROOT / "smoke-test"

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)


# -----------------------------
# LoRA
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
# Formatting
# -----------------------------

def format_example(example):
    return f"""### Instruction:
Summarize the following article as a short, engaging social media post.

### Article:
{example['blog']}

### Post:
{example['post']}"""


# -----------------------------
# Load only 5 examples
# -----------------------------

pairs = []

with DATASET_PATH.open(encoding="utf-8") as f:
    for line in f:
        pairs.append(json.loads(line))

pairs = pairs[:5]

print(f"Loaded {len(pairs)} examples for smoke test.")

texts = [format_example(p) for p in pairs]


# -----------------------------
# Tokenization
# -----------------------------

tokenized = tokenizer(
    texts,
    truncation=True,
    max_length=256,
    padding="max_length",
)

ds = Dataset.from_dict(tokenized)

ds = ds.map(
    lambda x: {"labels": x["input_ids"]}
)


# -----------------------------
# Training
# -----------------------------

args = TrainingArguments(
    output_dir=str(SMOKE_OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
)


print("\nStarting smoke test...\n")

trainer.train()


print("\nSmoke test passed — loss logged above, no errors.")
