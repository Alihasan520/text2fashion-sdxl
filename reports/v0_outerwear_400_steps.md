# V0 Outerwear Experiment: 400 Steps

## Purpose

This experiment validated the project workflow before training the main all-category LoRA model.

## Scope

- Category: outer
- Base model: stabilityai/stable-diffusion-xl-base-1.0
- Method: LoRA
- Resolution: 512
- Training steps: 400
- Checkpoint interval: 100 steps

## Saved Artifacts

- checkpoint-300
- checkpoint-400
- final LoRA weights
- training logs

## Role in the Project

This run is treated as a proof of pipeline. The main version is trained separately from the SDXL base model using the top, bottom, and outer categories.
