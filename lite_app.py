import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Utilise MPS si disponible, sinon CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Charge un modèle léger
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to(device)

# Désactive le safety checker (bug fréquent sur Mac)
pipe.safety_checker = lambda images, **kwargs: (images, False)

def generate_image(prompt):
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
    return image

gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=1, placeholder="Enter your prompt"),
    outputs="image",
    title="🧠 Stable Diffusion Lite (Mac/MPS)"
).launch()
