import os
from torchvision import transforms

from masking_inpainting import mask_and_inpaint
from VLinference import inference

if __name__ == "__main__":
    # Define the input image path and prompts
    input_image_path = "images/beret.jpg"
    mask_prompt = "red hat"
    inpaint_prompt = "blue hat"
    description_scale = 10
    output_image_path = "output_image_refined.jpg"

    # change image path to absolute path
    input_image_path = os.path.abspath(input_image_path)

    # Perform inference using the Qwen2.5-VL model
    prompt = f"Please describe in detail the {mask_prompt} in the image. Use at least {description_scale} adjective words, including its type, color, texture, etc. Do not use full sentences."
    adjectives = inference(input_image_path, prompt)

    # Perform masking and inpainting
    inpaint_prompt = f"{adjectives} {inpaint_prompt}"
    output_image = mask_and_inpaint(input_image_path, mask_prompt, inpaint_prompt, verbose=False)
    output_image.save(output_image_path)

    