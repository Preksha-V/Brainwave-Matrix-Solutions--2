from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float32
).to("cpu")  

prompt = "cute puppies"
image = pipeline(prompt).images[0]

image.save("generated_image.png")
print("Image generated and saved successfully!")
