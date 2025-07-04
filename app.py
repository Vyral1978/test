import spaces
import gradio as gr
import numpy as np
import PIL.Image
from PIL import Image
import random
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch
from compel import Compel, ReturnedEmbeddingsType

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Make sure to use torch.float16 consistently throughout the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "votepurchase/pornmasterPro_noobV3VAE",
    torch_dtype=torch.float32,
    variant="fp16",
    use_safetensors=True
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# Force all components to use the same dtype

# 追加: Initialize Compel for long prompt processing
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
    truncate_long_prompts=False
)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1216

# 追加: Simple long prompt processing function
def process_long_prompt(prompt, negative_prompt=""):
    """Simple long prompt processing using Compel"""
    try:
        conditioning, pooled = compel([prompt, negative_prompt])
        return conditioning, pooled
    except Exception as e:
        print(f"Long prompt processing failed: {e}, falling back to standard processing")
        return None, None
    
@spaces.GPU
def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    # 変更: Remove the 60-word limit warning and add long prompt check
    use_long_prompt = len(prompt.split()) > 60 or len(prompt) > 300
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    
    try:
        # 追加: Try long prompt processing first if prompt is long
        if use_long_prompt:
            print("Using long prompt processing...")
            conditioning, pooled = process_long_prompt(prompt, negative_prompt)
            
            if conditioning is not None:
                output_image = pipe(
                    prompt_embeds=conditioning[0:1],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]
                return output_image
        
        # Fall back to standard processing
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
                placeholder="Enter your prompt (long prompts are automatically supported)",  # 変更: Updated placeholder
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
