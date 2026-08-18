from datasets import load_dataset

raw = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:40]")
articles = [r["article"] for r in raw]

print(f"Loaded {len(articles)} articles")
print(articles[0][:500])