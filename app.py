import spaces
import gradio as gr
import numpy as np
import PIL.Image
from PIL import Image
import random
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure to use torch.float16 consistently throughout the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "votepurchase/pornmasterPro_noobV3VAE",
    torch_dtype=torch.float16,
    variant="fp16",  # Explicitly use fp16 variant
    use_safetensors=True  # Use safetensors if available
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# Force all components to use the same dtype
pipe.text_encoder.to(torch.float16)
pipe.text_encoder_2.to(torch.float16)
pipe.vae.to(torch.float16)
pipe.unet.to(torch.float16)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1216
    
@spaces.GPU
def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    # Check and truncate prompt if too long (CLIP can only handle 77 tokens)
    if len(prompt.split()) > 60:  # Rough estimate to avoid exceeding token limit
        print("Warning: Prompt may be too long and will be truncated by the model")
        
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    
    try:
        output_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        
        return output_image
    except RuntimeError as e:
        print(f"Error during generation: {e}")
        # Return a blank image with error message
        error_img = Image.new('RGB', (width, height), color=(0, 0, 0))
        return error_img


css = """
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt (keep it under 60 words for best results)",
                container=False,
            )

            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)
        
        with gr.Accordion("Advanced Settings", open=False):

            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                value="nsfw, (low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn"
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=20.0,
                    step=0.1,
                    value=7,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=28,
                    step=1,
                    value=28,
                )

    run_button.click(
        fn=infer,
        inputs=[prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result]
    )

demo.queue().launch()