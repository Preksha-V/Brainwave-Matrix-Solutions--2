from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float32  # Use float32 for CPU
).to("cpu")  # Force the model to use CPU

# Define your text prompt
prompt = "mango"

# Generate the image
image = pipeline(prompt).images[0]

# Save the image
image.save("generated_image.png")
print("Image generated and saved successfully!")
