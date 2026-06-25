# Batch 3 Contents

This batch adds inference, Hugging Face deployment utilities, Gradio Space files, and project reports.

## Included

- Reusable inference pipeline
- Single-image generation utility
- Hugging Face configuration reader
- Model card generator
- Hub upload script
- Best LoRA export script
- Gradio app for Hugging Face Spaces
- Report templates for v0 and v1

## Main Commands

Export the selected LoRA:

```bash
python scripts/export_best_lora.py --source outputs/checkpoints/checkpoint-3000 --target outputs/exports/best_lora
```

Create and push the model repo:

```bash
python src/text2fashion/hub/push_to_hub.py --config configs/hub.yaml
```

Run the Gradio app locally:

```bash
python app/app.py
```
