from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2fashion.evaluation.build_grids import build_prompt_grids
from text2fashion.evaluation.generate_samples import collect_lora_sources, generate_for_lora_sources, load_sdxl_pipeline
from text2fashion.evaluation.prompts import get_prompts
from text2fashion.evaluation.visual_score_template import create_visual_score_template
from text2fashion.settings import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA checkpoints with fixed prompts.")
    parser.add_argument("--config", default="configs/eval_kaggle.yaml", help="Evaluation config path.")
    parser.add_argument("--skip-generation", action="store_true", help="Only rebuild grids and score template.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    path_cfg = config["paths"]
    infer_cfg = config["inference"]
    eval_cfg = config["evaluation"]
    checkpoint_cfg = config.get("checkpoints", {})

    base_model = path_cfg.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
    lora_root = resolve_path(path_cfg["lora_root"])
    eval_output_dir = resolve_path(path_cfg["eval_output_dir"])
    score_template_path = resolve_path(path_cfg["score_template_path"])

    prompt_records = get_prompts(prompts_per_category=int(eval_cfg.get("prompts_per_category", 6)))
    seeds = [int(seed) for seed in eval_cfg.get("seeds", [42])]

    lora_sources = collect_lora_sources(
        lora_root=lora_root,
        include_final=bool(checkpoint_cfg.get("include_final", True)),
        names=checkpoint_cfg.get("names") or None,
    )

    if not lora_sources:
        raise FileNotFoundError(f"No LoRA sources found in {lora_root}")

    checkpoint_names = [name for name, _ in lora_sources]
    print("Evaluation sources:")
    for name, path in lora_sources:
        print(f"- {name}: {path}")

    if not args.skip_generation:
        pipe = load_sdxl_pipeline(base_model=base_model)
        generate_for_lora_sources(
            pipe=pipe,
            lora_sources=lora_sources,
            prompt_records=prompt_records,
            output_dir=eval_output_dir,
            seeds=seeds,
            num_inference_steps=int(infer_cfg.get("num_inference_steps", 30)),
            guidance_scale=float(infer_cfg.get("guidance_scale", 7.5)),
            height=int(infer_cfg.get("height", 768)),
            width=int(infer_cfg.get("width", 768)),
        )

    grid_dir = build_prompt_grids(
        eval_output_dir=eval_output_dir,
        prompt_records=prompt_records,
        checkpoint_names=checkpoint_names,
        seeds=seeds,
    )

    template_path = create_visual_score_template(
        output_path=score_template_path,
        checkpoint_names=checkpoint_names,
        prompt_records=prompt_records,
        seeds=seeds,
    )

    print(f"Saved grids to: {grid_dir}")
    print(f"Saved score template to: {template_path}")


if __name__ == "__main__":
    main()
