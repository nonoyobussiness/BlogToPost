import os
import json
import time

from datasets import load_dataset
from groq import Groq


# -----------------------------
# Phase 1: Load articles
# -----------------------------

raw = load_dataset(
    "abisee/cnn_dailymail",
    "3.0.0",
    split="train[:40]"
)

articles = [r["article"] for r in raw]

print(f"Loaded {len(articles)} articles")


# -----------------------------
# Phase 2: Generate posts
# -----------------------------

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

SYSTEM_PROMPT = """You are a social media editor.

Given a news article, write ONE short LinkedIn-style post.

Rules:
- Write exactly 2-3 sentences.
- No hashtags.
- No emojis.
- Capture the single most important point.
- Do not add information that is not present in the article.
- Do not say "According to the article".
- Do not introduce the post with phrases like "Here's a summary".
- Output ONLY the post.
"""


pairs = []

for i, article in enumerate(articles):
    print(f"\nGenerating {i + 1}/{len(articles)}...")

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": f"""You are a social media editor.

Given the following news article, write ONE short LinkedIn-style post.

Rules:
- Write exactly 2-3 sentences.
- No hashtags.
- No emojis.
- Capture the single most important point.
- Do not add information that is not present in the article.
- Output ONLY the final post.

Article:
{article[:3000]}

Write the post:"""
                }
            ],

            temperature=0.4,
            max_completion_tokens=512,
            reasoning_effort="low",
            include_reasoning=False,
        )

        post = (response.choices[0].message.content or "").strip()

        if not post:
            print("  -> Empty response, skipping")
            continue

        pairs.append({
            "blog": article,
            "post": post
        })

        print(f"  -> {post}")

        time.sleep(2.5)

    except Exception as e:
        print(f"  -> ERROR: {e}")

# -----------------------------
# Save dataset
# -----------------------------

with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


print(f"\nDone!")
print(f"Saved {len(pairs)} pairs to dataset.jsonl")