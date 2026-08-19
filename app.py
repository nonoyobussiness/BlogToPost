import re
from pathlib import Path

import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from article_extractor import ArticleExtractionError, extract_article


# -----------------------------
# Configuration
# -----------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REPO_ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = str(REPO_ROOT / "model" / "blog-to-post-lora-v2")

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 80
MAX_ARTICLE_CHARS_FOR_MODEL = 2000
PREVIEW_CHARS = 1500


# -----------------------------
# Load model
# -----------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Model loaded successfully.")


# -----------------------------
# Generate post
# -----------------------------

def clean_generated_post(post):
    post = post.strip()

    # Prevent template leakage without rewriting model wording.
    for marker in ("### Instruction:", "### Article:", "### Post:"):
        if marker in post:
            post = post.split(marker)[0].strip()

    # Remove clear hashtag tokens like "#creativity" while preserving words like "C#".
    post = re.sub(r"(^|\s)#[A-Za-z0-9_]+\b", r"\1", post)
    post = re.sub(r"(^|\s)#(?=\s|$)", r"\1", post)

    # Keep sentence boundaries intact while removing excessive whitespace.
    post = re.sub(r"[ \t]+", " ", post)
    post = re.sub(r" *\n *", "\n", post)
    post = re.sub(r"\n{3,}", "\n\n", post)

    return post.strip()


def generate_post(article):

    if not article or not article.strip():
        return "Please enter an article."

    prompt = f"""### Instruction:
Summarize the following article as a short, engaging social media post.

### Article:
{article[:MAX_ARTICLE_CHARS_FOR_MODEL]}

### Post:
"""

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

    # Get only newly generated tokens
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    post = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return clean_generated_post(post)


def build_preview(article):
    if len(article) <= PREVIEW_CHARS:
        return article

    return article[:PREVIEW_CHARS].rstrip() + "\n\n[Preview truncated]"


def generate_from_url(url, progress=gr.Progress(track_tqdm=False)):
    try:
        progress(0.15, desc="Fetching URL...")
        article = extract_article(url)

        progress(0.55, desc="Article extracted.")
        preview = build_preview(article)

        progress(0.75, desc="Generating social post...")
        post = generate_post(article)

        progress(1.0, desc="Done.")
        return preview, post

    except ArticleExtractionError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error("Something went wrong while generating the post.") from exc


# -----------------------------
# Gradio UI
# -----------------------------

demo = gr.Interface(
    fn=generate_from_url,

    inputs=gr.Textbox(
        lines=1,
        placeholder="https://example.com/article",
        label="Article URL",
    ),

    outputs=[
        gr.Textbox(
            lines=12,
            label="Extracted Article Preview",
        ),
        gr.Textbox(
            lines=6,
            label="Generated Social Post",
        ),
    ],

    title="Article URL to Social Post",

    description=(
        "A locally deployed Qwen2.5-0.5B model fine-tuned "
        "with LoRA to convert article URLs into short social-media posts."
    ),
    submit_btn="Generate Social Post",
)


# -----------------------------
# Launch
# -----------------------------

if __name__ == "__main__":
    demo.launch()
