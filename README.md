# BlogToPost

BlogToPost turns an article URL into a short social-media post using local article extraction and a locally fine-tuned Qwen model.

## Overview

The app accepts an article URL, extracts the main article text, and sends the cleaned text to a LoRA-adapted `Qwen/Qwen2.5-0.5B-Instruct` model. Inference runs locally after the webpage has been fetched.

```
URL -> article extraction -> locally fine-tuned Qwen -> social media post
```

## Architecture

```
User URL
  |
  v
Article extraction
  |
  v
Clean article text
  |
  v
Qwen2.5-0.5B-Instruct + LoRA
  |
  v
Output cleanup
  |
  v
Social media post
```

The Gradio app validates the URL, allows only `http://` and `https://`, extracts article text with `trafilatura`, truncates long articles before model inference, generates deterministically on CPU, and removes lightweight output artifacts such as leaked prompt headers or hashtags.

## Why This Model?

This project was designed around a CPU-only development environment with about 8 GB RAM. `Qwen/Qwen2.5-0.5B-Instruct` was selected because it is small enough to run locally while still supporting instruction-style prompting. A larger model would likely produce stronger summaries, but would be less practical for this hardware constraint.

## Fine-Tuning

The fine-tuned adapter in this repository is a PEFT LoRA adapter for:

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA rank: `8`
- LoRA alpha: `16`
- Target modules: `q_proj`, `v_proj`
- LoRA dropout: `0.05`
- Learning rate: `2e-4`
- Epochs: `3`
- Dataset size: small development dataset

These settings are defined in [scripts/train.py](scripts/train.py) and the saved adapter metadata in [model/blog-to-post-lora-v2/adapter_config.json](model/blog-to-post-lora-v2/adapter_config.json).

The repository does not include a Colab notebook or training run log, so this README documents the verifiable training configuration present in the project files.

## Dataset

The dataset is stored at [data/dataset.jsonl](data/dataset.jsonl). Each row contains:

- `blog`: source article text
- `post`: reference social-media post

The dataset generation script uses CNN/DailyMail-style articles from `abisee/cnn_dailymail` and generates reference posts during development with Groq using `openai/gpt-oss-20b`. Groq was used only to bootstrap training targets in [scripts/generate_dataset.py](scripts/generate_dataset.py). It is not part of inference or deployment.

Before redistributing or publishing the dataset, review the licensing and redistribution terms for the source article dataset and any generated derivatives. This repository does not make a legal claim about redistribution rights.

## Evaluation

[scripts/evaluate.py](scripts/evaluate.py) compares the base Qwen model against the LoRA-adapted model on the first five dataset examples. The project notes define three qualitative checks:

- Length: whether the output is social-post length
- Faithfulness: whether it avoids adding unsupported details
- Style: whether it reads like a social-media post rather than a generic summary

No saved benchmark scores or judge outputs are included in the repository, so no numeric evaluation results are claimed here.

Known limitations:

- Tiny development dataset
- 0.5B parameter base model
- Occasional hallucinations
- Occasional style or prompt-template leakage
- Long articles are truncated before inference
- Proof-of-concept summarization quality, not production-grade summarization

## Running Locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the Gradio app:

```powershell
python app.py
```

Open the local URL printed by Gradio, usually:

```text
http://127.0.0.1:7860
```

The application accepts an article URL, extracts the article text, and returns a generated social-media post.

## Scripts

```powershell
python scripts/generate_dataset.py
python scripts/smoke_test.py
python scripts/train.py
python scripts/evaluate.py
python scripts/verify_cpu.py
```

Do not run training scripts unless you intend to regenerate the adapter. The app uses the checked-in adapter under `model/blog-to-post-lora-v2/`.

## MLOps / Engineering

This is a small local ML project, not a production MLOps platform. It includes:

- Reproducible Python dependencies in `requirements.txt`
- Versioned LoRA adapter artifact under `model/`
- Dataset generation script
- Training script
- Evaluation script
- Smoke-test training script
- Automated tests for URL validation and extraction failure handling
- GitHub Actions CI for lightweight tests

GitHub Actions intentionally does not run Qwen inference or download the base model.

## Deployment

The current deployment target is local Gradio:

```powershell
python app.py
```

### Future Deployment

A future version could be deployed as a Hugging Face Space or packaged behind a small API. For a more efficient local runtime, the adapter could also be merged and converted to GGUF/Ollama, but that is not implemented in this repository.

## Project Structure

```text
BlogToPost/
|-- app.py
|-- article_extractor.py
|-- scripts/
|   |-- generate_dataset.py
|   |-- train.py
|   |-- evaluate.py
|   |-- smoke_test.py
|   `-- verify_cpu.py
|-- tests/
|   `-- test_article_extractor.py
|-- data/
|   `-- dataset.jsonl
|-- model/
|   `-- blog-to-post-lora-v2/
|-- .github/
|   `-- workflows/
|       `-- tests.yml
|-- README.md
|-- requirements.txt
|-- .gitignore
`-- plan.md
```

## Limitations and Future Work

- Train on a larger, better-filtered dataset
- Improve target-quality filtering before training
- Use a longer-context model or chunking strategy for long articles
- Add stronger evaluation beyond manual qualitative comparison
- Explore GGUF/Ollama for faster local inference
- Add public deployment after validating model behavior
- Improve article extraction for blocked pages, paywalls, and unusual layouts
