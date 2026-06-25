# Batch 1 Contents

This batch provides the project structure and the full data preparation pipeline.

## Included

- Base repository files
- Safe Hugging Face configuration templates
- Data configuration for all categories
- Caption cleaning utilities
- Dataset splitting utilities
- Diffusers folder export utility
- Validation utilities
- Data preparation script

## Main Command

```bash
python scripts/prepare_all_data.py --config configs/data_all.yaml
```

## Not Included Yet

The following will be added in later batches:

- Training command builders
- Resume utilities
- Checkpoint evaluation
- Gradio app
- Hugging Face model card and upload scripts
