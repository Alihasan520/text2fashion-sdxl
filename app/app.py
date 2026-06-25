from __future__ import annotations

import os

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
LORA_REPO_ID = os.getenv("LORA_REPO_ID", "your-hf-username/text2fashion-sdxl-lora")

DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, bad product photo, messy background"

EXAMPLES = [
    "black puffer jacket with hood, studio product photo, minimal background",
    "white t-shirt with clean minimal design, studio product photo, minimal background",
    "blue straight fit jeans with front pockets, studio product photo, minimal background",
    "beige coat with belt and high collar, studio product photo, minimal background",
    "black cargo pants with side pockets, studio product photo, minimal background",
]

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.load_lora_weights(LORA_REPO_ID)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.set_progress_bar_config(disable=True)


def generate(
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
):
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        height=int(height),
        width=int(width),
        generator=generator,
    ).images[0]
    return image


with gr.Blocks(title="Text2Fashion-SDXL") as demo:
    gr.Markdown("# Text2Fashion-SDXL")
    gr.Markdown("Generate fashion product images using SDXL + LoRA.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0], lines=3)
            negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)
            seed = gr.Number(label="Seed", value=42, precision=0)
            steps = gr.Slider(label="Inference steps", minimum=15, maximum=50, value=30, step=1)
            guidance_scale = gr.Slider(label="Guidance scale", minimum=3.0, maximum=12.0, value=7.5, step=0.5)
            height = gr.Dropdown(label="Height", choices=[512, 768, 1024], value=768)
            width = gr.Dropdown(label="Width", choices=[512, 768, 1024], value=768)
            button = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(label="Generated image", type="pil")

    gr.Examples(examples=[[example] for example in EXAMPLES], inputs=[prompt])

    button.click(
        fn=generate,
        inputs=[prompt, negative_prompt, seed, steps, guidance_scale, height, width],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
