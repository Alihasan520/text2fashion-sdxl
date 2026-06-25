# Batch 2 Contents

This batch adds the training, resume, checkpoint evaluation, and visual scoring workflow.

## Included

- Kaggle data configuration for all categories
- Training command builder
- Resume and checkpoint utilities
- Evaluation prompt suite
- Checkpoint sample generation
- Comparison grid generation
- Visual scoring template
- Single Kaggle notebook for training and evaluation
- Training and evaluation scripts

## Main Commands

Prepare data for Kaggle:

```bash
python scripts/prepare_all_data.py --config configs/data_all_kaggle.yaml
```

Train a new LoRA from SDXL base:

```bash
python scripts/train_all_lora.py --config configs/train_all_kaggle.yaml --download-script
```

Clean incomplete outputs before a fresh run if needed:

```bash
python scripts/train_all_lora.py --config configs/train_all_kaggle.yaml --download-script --clean-output
```

Resume training:

```bash
python scripts/resume_all_lora.py --config configs/resume_all_kaggle.yaml
```

Evaluate checkpoints:

```bash
python scripts/evaluate_checkpoints.py --config configs/eval_kaggle.yaml
```
