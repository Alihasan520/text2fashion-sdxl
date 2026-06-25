# Text2Fashion-SDXL

Text2Fashion-SDXL is an end-to-end text-to-image portfolio project for generating fashion product images using **Stable Diffusion XL (SDXL)** and **LoRA fine-tuning**.

The project is organized as a professional machine learning repository with separate modules for data preparation, training, checkpoint evaluation, inference, Hugging Face deployment, and a Gradio interface.

## Project Overview

The main version trains a new SDXL LoRA from the SDXL base model on three fashion product categories:

- `top`
- `bottom`
- `outer`

The dataset is loaded from Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("hahminlew/kream-product-blip-captions")
```

An earlier `outer`-only experiment trained for 400 steps is documented as a pipeline validation run. It is not used as the starting point for the main all-category model.

## Goals

This repository provides:

- Reproducible data preparation for fashion product captions
- Clean caption normalization for text-to-image training
- Diffusers-compatible image folder export with `metadata.jsonl`
- Resumable SDXL LoRA training on Kaggle
- Checkpoint-based visual evaluation
- Manual visual scoring templates
- Inference utilities for local testing
- Hugging Face Model Hub configuration
- Gradio interface for Hugging Face Spaces

## Project Structure

```text
text2fashion-sdxl/
│
├── configs/
│   ├── data_all.yaml
│   ├── data_all_kaggle.yaml
│   ├── train_all_kaggle.yaml
│   ├── resume_all_kaggle.yaml
│   ├── eval.yaml
│   ├── eval_kaggle.yaml
│   ├── inference.yaml
│   └── hub.example.yaml
│
├── notebooks/
│   └── 01_train_eval_kaggle.ipynb
│
├── src/text2fashion/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── hub/
│
├── scripts/
│   ├── prepare_all_data.py
│   ├── train_all_lora.py
│   ├── resume_all_lora.py
│   ├── evaluate_checkpoints.py
│   └── export_best_lora.py
│
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── examples/
│
├── reports/
├── outputs/
├── data/
├── README.md
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For Kaggle training, the notebook can also install the required packages inside the runtime.

## Hugging Face Configuration

This project uses safe local configuration for Hugging Face deployment.

Copy the example files:

```text
configs/hub.example.yaml -> configs/hub.yaml
.env.example -> .env
```

Then edit the local files with your Hugging Face username, repository names, and token.

The following local files are intentionally ignored by Git:

```text
.env
configs/hub.yaml
```

This keeps tokens and account-specific deployment settings outside GitHub.

## Prepare Data

Prepare the all-category dataset:

```bash
python scripts/prepare_all_data.py --config configs/data_all.yaml
```

For Kaggle paths:

```bash
python scripts/prepare_all_data.py --config configs/data_all_kaggle.yaml
```

The preparation pipeline creates:

- cleaned captions
- train / validation / test splits
- exported training images
- `metadata.jsonl`
- validation summary

## Train LoRA

Train a new all-category LoRA from SDXL base:

```bash
python scripts/train_all_lora.py --config configs/train_all_kaggle.yaml --download-script
```

If the output directory contains incomplete files from a failed run and no valid checkpoints, use:

```bash
python scripts/train_all_lora.py --config configs/train_all_kaggle.yaml --download-script --clean-output
```

## Resume Training

Resume from the latest checkpoint:

```bash
python scripts/resume_all_lora.py --config configs/resume_all_kaggle.yaml
```

In step-based training, `max_train_steps` should be the final total target step count. For example, to continue from step 1500 to step 3000, set `max_train_steps: 3000`.

## Evaluate Checkpoints

Generate samples and comparison grids:

```bash
python scripts/evaluate_checkpoints.py --config configs/eval_kaggle.yaml
```

The evaluation workflow uses:

- fixed prompts
- fixed seeds
- checkpoint comparison grids
- visual scoring templates

Manual evaluation focuses on:

- prompt alignment
- category correctness
- color correctness
- visual quality
- artifact control
- full product view correctness

## Inference

The inference utilities load SDXL and a selected LoRA checkpoint using Diffusers. These utilities are used by both local testing scripts and the Gradio app.

## Gradio App

The Hugging Face Space entry point is:

```text
app/app.py
```

The Space dependencies are defined in:

```text
app/requirements.txt
```

Before deployment, set the LoRA repository or path through configuration or Space environment variables.

## Large Files

Do not commit large artifacts to GitHub:

- raw datasets
- processed images
- checkpoints
- model weights
- generated outputs

Use Kaggle Datasets or Hugging Face Hub for large files.
