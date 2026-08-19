import json
from pathlib import Path

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -----------------------------
# Configuration
# -----------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "dataset.jsonl"
ADAPTER_PATH = str(REPO_ROOT / "model" / "blog-to-post-lora-v2")

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 80


# -----------------------------
# Load tokenizer
# -----------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token


# -----------------------------
# Load BASE model
# -----------------------------

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
)


# -----------------------------
# Load SEPARATE model for LoRA
# -----------------------------

print("Loading fine-tuned model...")

tuned_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
)

tuned_model = PeftModel.from_pretrained(
    tuned_base,
    ADAPTER_PATH
)

print("Models loaded.")


# -----------------------------
# Prompt
# -----------------------------

def build_prompt(article):
    return f"""### Instruction:
Summarize the following article as a short, engaging social media post.

### Article:
{article[:2000]}

### Post:
"""


# -----------------------------
# Generation
# -----------------------------

def generate(model, article):

    prompt = build_prompt(article)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id,
        )

    # Only take newly generated tokens
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return result


# -----------------------------
# Load dataset
# -----------------------------

pairs = []

with DATASET_PATH.open(encoding="utf-8") as f:
    for line in f:
        pairs.append(json.loads(line))


print(f"Loaded {len(pairs)} dataset examples.")


# -----------------------------
# Evaluation examples
# -----------------------------

evaluation_pairs = pairs[:5]


# -----------------------------
# Compare
# -----------------------------

for i, pair in enumerate(evaluation_pairs, start=1):

    article = pair["blog"]

    print("\n" + "=" * 80)
    print(f"EVALUATION EXAMPLE {i}")
    print("=" * 80)

    print("\nARTICLE:")
    print(article[:1000])
    print("...")

    print("\nBASE QWEN:")
    base_output = generate(
        base_model,
        article
    )
    print(base_output)

    print("\nFINE-TUNED QWEN:")
    tuned_output = generate(
        tuned_model,
        article
    )
    print(tuned_output)

    print("\nREFERENCE POST:")
    print(pair["post"])


print("\nEvaluation complete.")
