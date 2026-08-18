# 📱 Blog → Social Post Fine-Tune
### Project Bible — CPU-only laptop, same-day build

> **Hardware:** i5-1235U, 8 GB RAM, no dedicated GPU (Intel UHD, no CUDA). Every decision below is made because of that constraint, not in spite of it.
> **Goal:** Fine-tune a small instruction model to turn a blog post / article into a short social media post, and deploy it locally.
> **Time budget:** 4–6 hours total. If you're behind schedule, cut in this order: dataset size → epochs → skip GGUF export (never skip the smoke test).

---

## Table of Contents

1. [What You're Building](#1-what-youre-building)
2. [Hardware-Driven Decisions](#2-hardware-driven-decisions)
3. [Environment Setup](#3-environment-setup)
4. [Phase 1 — Dataset Sourcing](#phase-1--dataset-sourcing)
5. [Phase 2 — Reference Post Generation (Prompt)](#phase-2--reference-post-generation-prompt)
6. [Phase 3 — Instruction Formatting (Prompt Template)](#phase-3--instruction-formatting-prompt-template)
7. [Phase 4 — Smoke Test](#phase-4--smoke-test)
8. [Phase 5 — Full LoRA Training](#phase-5--full-lora-training)
9. [Phase 6 — Evaluation (Prompt)](#phase-6--evaluation-prompt)
10. [Phase 7 — Local Deployment](#phase-7--local-deployment)
11. [Phase 8 — README / Documentation](#phase-8--readme--documentation)
12. [Stretch Goals (only if time remains)](#stretch-goals)
13. [Common Errors and Fixes](#common-errors-and-fixes)
14. [Repository Structure](#repository-structure)

---

## 1. What You're Building

```
INPUT:   A blog post / article (long-form text)
OUTPUT:  A short social media post (LinkedIn/Twitter-style) capturing the key point

Pipeline:
  Blog text
     │
     ▼
  Fine-tuned small LLM (Qwen2.5-0.5B-Instruct or SmolLM2-360M, LoRA)
     │
     ▼
  Generated social post
```

**Why this task:** it's a clean instruction-tuning problem (long input → short styled output), easy to eyeball-evaluate, and doesn't require a specialized domain dataset you'd have to source carefully.

---

## 2. Hardware-Driven Decisions

| Decision | Choice | Why |
|---|---|---|
| Base model | `Qwen2.5-0.5B-Instruct` (fallback: `SmolLM2-360M-Instruct`) | Small enough for real LoRA training on 8 GB RAM, CPU-only, in under 2 hours |
| Quantization for training | **None** — plain fp32/bf16 LoRA | `bitsandbytes` 4-bit needs CUDA, which this machine doesn't have |
| Dataset size | 30–50 pairs | Enough for LoRA to pick up a style shift; more won't fit the time budget |
| LoRA rank | 8–16 | Higher rank = more compute for negligible gain at this dataset size |
| Deployment (today) | PEFT adapter + Gradio, loaded directly | GGUF conversion + quantization is a real time sink — save for stretch goals |
| Deployment (later) | Merge → GGUF → Ollama | Matches the pattern used in the nldb project; do this once the core project works |

---

## 3. Environment Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install torch transformers peft datasets accelerate gradio
```

Deliberately **not installed**: `trl`, `bitsandbytes` — you don't need either for a plain CPU LoRA loop, and `bitsandbytes` will fail to initialize CUDA on this machine anyway.

Verify CPU-only setup:
```python
import torch
print(torch.cuda.is_available())  # should print False — that's expected, not an error
```

---

## Phase 1 — Dataset Sourcing

Don't scrape — pull real articles instantly from an existing dataset as your "blog" stand-in:

```python
from datasets import load_dataset

raw = load_dataset("cnn_dailymail", "3.0.0", split="train[:40]")
articles = [r["article"] for r in raw]
```

40 articles gives you room to drop a few bad generations and still land at 30–35 usable pairs.

---

## Phase 2 — Reference Post Generation (Prompt)

This step is **offline, development-time only** — it never runs inside your deployed model. You're using a larger hosted model once, here, to bootstrap training targets, the same pattern used in the Redrob project (Groq used offline for JD analysis, never at inference time).

Use Groq's free tier (fast) or any API you have access to. Prompt template, one call per article:

```
SYSTEM PROMPT:
You are a social media editor. Given a news article, write ONE short
LinkedIn-style post (2–3 sentences, no hashtags, no emojis) that captures
the single most important point. Do not add information not in the article.

USER PROMPT:
Article:
{article_text}

Write the post:
```

Generation script sketch:

```python
import os, json, time
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are a social media editor. Given a news article, write ONE short
LinkedIn-style post (2-3 sentences, no hashtags, no emojis) that captures
the single most important point. Do not add information not in the article."""

pairs = []
for article in articles:
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Article:\n{article[:3000]}\n\nWrite the post:"},
        ],
        temperature=0.4,
    )
    post = resp.choices[0].message.content.strip()
    pairs.append({"blog": article, "post": post})
    time.sleep(0.5)  # stay under rate limits

with open("dataset.jsonl", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
```

Skim the outputs before moving on — drop any that are empty, off-topic, or absurdly long. You want 30–40 clean pairs left.

---

## Phase 3 — Instruction Formatting (Prompt Template)

This is the template baked into every training example — the model learns to complete *after* this exact structure, so use the same template at inference time:

```python
def format_example(example):
    return f"""### Instruction:
Summarize the following article as a short, engaging social media post.

### Article:
{example['blog']}

### Post:
{example['post']}"""
```

For inference later, you'll use the same header but stop before `### Post:` and let the model generate the completion.

---

## Phase 4 — Smoke Test

**Do not skip this.** Train on 5 examples, 1 epoch, and confirm the loop runs end-to-end before committing to the full run — this catches formatting/tokenizer bugs in 5 minutes instead of after a 2-hour wasted run.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

pairs = [json.loads(l) for l in open("dataset.jsonl")][:5]  # smoke test subset
texts = [format_example(p) for p in pairs]
tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
ds = Dataset.from_dict(tokenized)
ds = ds.map(lambda x: {"labels": x["input_ids"]})

args = TrainingArguments(
    output_dir="./smoke-test", num_train_epochs=1,
    per_device_train_batch_size=1, logging_steps=1,
    save_strategy="no", report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
print("Smoke test passed — loss logged above, no errors.")
```

If this runs clean, move to Phase 5. If it errors, fix it here — don't debug during the real run.

---

## Phase 5 — Full LoRA Training

Same script as Phase 4, but:
- Full dataset (30–50 examples) instead of 5
- 3 epochs instead of 1
- `output_dir="./blog-to-post-lora"`

```python
pairs = [json.loads(l) for l in open("dataset.jsonl")]  # full set
texts = [format_example(p) for p in pairs]
tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
ds = Dataset.from_dict(tokenized)
ds = ds.map(lambda x: {"labels": x["input_ids"]})

args = TrainingArguments(
    output_dir="./blog-to-post-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.save_model()
```

**Checkpoint at 20 minutes:** if progress suggests this will run past ~2 hours, kill it and either drop to 2 epochs or trim the dataset to 25 examples. A shorter, finished run beats a longer, abandoned one.

While this runs in the background, move on to writing Phase 8 (the README) — don't sit and watch the loss curve.

---

## Phase 6 — Evaluation (Prompt)

Hold out 3–5 articles the model never trained on. Generate with both the base model and the fine-tuned model, then compare.

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(model_id)
tuned = PeftModel.from_pretrained(base, "./blog-to-post-lora")

def generate(model, article):
    prompt = f"""### Instruction:
Summarize the following article as a short, engaging social media post.

### Article:
{article}

### Post:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**inputs, max_new_tokens=80, temperature=0.7, do_sample=True)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("### Post:")[-1].strip()
```

Score each pair on 3 criteria (write these down, don't just eyeball it):
1. **Length fit** — is it actually social-post length, or does it ramble like the base model?
2. **Faithfulness** — does it avoid inventing details not in the article?
3. **Style** — does it read as a social post rather than a summary paragraph?

**Optional LLM-as-judge prompt**, if you want a scored comparison instead of just reading them yourself:

```
You are evaluating two AI-generated social media posts summarizing the same article.

Article: {article}
Post A: {base_output}
Post B: {tuned_output}

Rate each post 1-5 on: length appropriateness, faithfulness to the article,
and social-media tone. Return your answer as JSON: {"post_a": {...}, "post_b": {...}}
```

(This judge call is dev-time only, same as Phase 2 — never part of the deployed pipeline.)

---

## Phase 7 — Local Deployment

Skip GGUF/Ollama today — load the adapter directly for a fast working demo:

```python
import gradio as gr

def respond(article):
    return generate(tuned, article)

demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(lines=10, label="Blog / article text"),
    outputs=gr.Textbox(label="Generated social post"),
    title="Blog → Social Post (fine-tuned Qwen2.5-0.5B, CPU)",
)
demo.launch()
```

This satisfies "runs and deploys on my PC" for today's submission — everything after this point is optional.

---

## Phase 8 — README / Documentation

Write this **while Phase 5 trains in the background**, not after. Cover:

- Why this model size and this approach (hardware-driven decision, not a limitation you're hiding)
- How the dataset was built — real articles + LLM-generated reference posts, generation done offline/dev-time only
- The instruction template used for training and inference
- Before/after examples from your evaluation, with your 3 scoring criteria
- What you'd add with more time: bigger dataset, GGUF export + Ollama deployment, hyperparameter sweep

This section is what actually gets read in an interview — spend real time on it, not just a placeholder.

---

## Stretch Goals

Only attempt these if the core project (Phases 1–8) is done and working:

1. **Merge + GGUF export + Ollama deployment** — same pipeline discussed for the Mistral fine-tune project, just applied to this smaller model. Much faster here since the model is <1B params instead of 7B.
2. **Bigger dataset** (100+ pairs) for a more convincing style shift.
3. **LoRA rank/alpha sweep** — document how output quality changes with rank 4 vs. 8 vs. 16.

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `CUDA not available` warnings | Expected and harmless — you're intentionally CPU-only |
| Training extremely slow (>10 min/epoch on 30 examples) | Reduce `max_length` to 256, confirm `per_device_train_batch_size=1` |
| Out-of-memory / process killed | Close other apps, reduce `max_length`, reduce dataset size |
| Generated posts are empty or repeat the prompt | Check the `### Post:` split logic in `generate()`, and confirm `pad_token` is set |
| Groq rate limit errors in Phase 2 | Add longer `time.sleep()`, or batch fewer articles per run |

---

## Repository Structure

```
blog-to-social-post/
├── plan.md                    # this file
├── README.md                  # write in Phase 8
├── requirements.txt
├── generate_dataset.py        # Phase 1 + 2
├── dataset.jsonl              # output of Phase 2
├── smoke_test.py              # Phase 4
├── train.py                   # Phase 5
├── evaluate.py                # Phase 6
├── app.py                     # Phase 7 — Gradio demo
└── blog-to-post-lora/         # saved LoRA adapter (output of Phase 5)
```