# Text2Fashion-SDXL

**Text2Fashion-SDXL** is an end-to-end AI fashion image generation project that uses **Stable Diffusion XL (SDXL)** with **LoRA fine-tuning** to generate fashion product images from text prompts.

The project covers the full machine learning workflow: dataset preparation, caption cleaning, SDXL LoRA training, checkpoint evaluation, inference, and web deployment through a lightweight Gradio-based interface.

## Live Demo

The deployed web interface is available here:

```text
https://winter-night-c8d8.spark724spark.workers.dev/
```

> Important: The GPU server must be started first before using the web interface. If the GPU runtime is not running, the website may load but image generation will not work correctly.

## GPU Server Notebook

Start the GPU inference server from the following Google Colab notebook:

```text
https://colab.research.google.com/drive/
```

Recommended workflow:

1. Open the Colab notebook.
2. Enable GPU runtime from Colab settings.
3. Run the notebook cells to start the inference server.
4. Copy the generated public server URL if the notebook provides one.
5. Make sure the frontend is connected to the active GPU server.
6. Open the live demo and generate fashion images from text prompts.

## Project Overview

The project fine-tunes **Stable Diffusion XL** using **LoRA** on fashion product data. The goal is to generate clean, product-focused fashion images based on structured natural-language prompts.

The model is designed to support fashion-related categories such as:

- Tops
- Bottoms
- Outerwear
- General fashion product images

The training pipeline uses fashion product captions and prepares them in a format compatible with the Hugging Face Diffusers ecosystem.

## Main Features

- Text-to-image fashion product generation
- SDXL-based image generation pipeline
- LoRA fine-tuning for fashion-specific visual style
- Caption cleaning and normalization
- Dataset preparation for Diffusers training
- Kaggle/Colab-compatible training workflow
- Checkpoint evaluation using fixed prompts and seeds
- Inference pipeline for generating images from custom prompts
- Web interface for interactive image generation
- Deployment-ready project structure

## Technical Stack

| Area | Technologies |
|---|---|
| Core Model | Stable Diffusion XL |
| Fine-tuning | LoRA |
| ML Framework | PyTorch, Hugging Face Diffusers |
| Data Processing | Python, Pandas, Datasets |
| Training Environment | Kaggle / Google Colab GPU |
| Interface | Gradio |
| Deployment | Web frontend + GPU inference server |
| Configuration | YAML, environment variables |

## Project Structure

```text
text2fashion-sdxl/
│
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── examples/
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
├── data/
│   ├── raw/
│   ├── processed/
│   ├── metadata/
│   └── samples/
│
├── notebooks/
│   └── 01_train_eval_kaggle.ipynb
│
├── outputs/
│
├── reports/
│   ├── selected_checkpoint.json
│   ├── visual_evaluation_template.csv
│   └── training / evaluation reports
│
├── scripts/
│   ├── prepare_all_data.py
│   ├── train_all_lora.py
│   ├── resume_all_lora.py
│   ├── evaluate_checkpoints.py
│   └── export_best_lora.py
│
├── src/text2fashion/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── hub/
│
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## Dataset

The project uses a fashion product caption dataset from Hugging Face:

```python
from datasets import load_dataset

subset = load_dataset("hahminlew/kream-product-blip-captions")
```

The data preparation pipeline performs:

- Dataset loading
- Caption normalization
- Product category filtering
- Train/validation/test splitting
- Image export
- `metadata.jsonl` generation for Diffusers training

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For the Gradio app only:

```bash
pip install -r app/requirements.txt
```

## Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

If Hugging Face Hub integration is needed, copy the Hub configuration example:

```bash
cp configs/hub.example.yaml configs/hub.yaml
```

Then add the required local values such as repository names or access tokens.

> Do not commit `.env`, access tokens, model weights, or private configuration files to GitHub.

## Prepare Data

Prepare the dataset locally:

```bash
python scripts/prepare_all_data.py --config configs/data_all.yaml
```

Prepare the dataset using Kaggle-compatible paths:

```bash
python scripts/prepare_all_data.py --config configs/data_all_kaggle.yaml
```

## Train the LoRA Model

Train the SDXL LoRA model:

```bash
python scripts/train_all_lora.py --config configs/train_all_kaggle.yaml --download-script
```

If the output folder contains incomplete files from a failed run, clean it before restarting:

```bash
python scripts/train_all_lora.py --config configs/train_all_kaggle.yaml --download-script --clean-output
```

## Resume Training

Resume training from the latest valid checkpoint:

```bash
python scripts/resume_all_lora.py --config configs/resume_all_kaggle.yaml
```

When using step-based training, `max_train_steps` should represent the final target number of steps, not only the additional steps.

## Evaluate Checkpoints

Generate evaluation samples from saved checkpoints:

```bash
python scripts/evaluate_checkpoints.py --config configs/eval_kaggle.yaml
```

The evaluation workflow uses:

- Fixed prompts
- Fixed random seeds
- Visual comparison grids
- Manual scoring templates
- Category and color alignment checks
- Artifact and quality review

## Inference

The inference module loads the SDXL base model and the selected LoRA weights to generate fashion product images from custom text prompts.

Typical inference flow:

1. Load SDXL base model.
2. Load selected LoRA checkpoint.
3. Enter a fashion-related prompt.
4. Generate and save the image output.

## Web App

The project includes a Gradio-based application entry point:

```text
app/app.py
```

Run locally:

```bash
python app/app.py
```

For the deployed version, start the GPU server first, then open:

```text
https://winter-night-c8d8.spark724spark.workers.dev/
```

## Example Prompts

```text
A studio product photo of a black oversized hoodie, minimal background, high detail, realistic fabric texture
```

```text
A clean catalog image of blue denim jeans, front view, white background, realistic lighting
```

```text
A beige winter coat, full product view, fashion catalog style, sharp details, neutral background
```

