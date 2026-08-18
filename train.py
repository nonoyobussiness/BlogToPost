import json


def format_example(example):
    return f"""### Instruction:
Summarize the following article as a short, engaging social media post.

### Article:
{example['blog']}

### Post:
{example['post']}"""


with open("dataset.jsonl", encoding="utf-8") as f:
    example = json.loads(next(f))


print(format_example(example))