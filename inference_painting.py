import os
from torchvision import transforms

from masking_inpainting import mask_and_inpaint
from VLinference import inference

def do_inf_inpaint(image_path, mask_prompt, inpaint_prompt, description_scale=10, mask=None):
    """
    Perform inference and inpainting on the input image.

    Args:
        image_path (str): Path to the input image.
        mask_prompt (str): Prompt for masking.
        inpaint_prompt (str): Prompt for inpainting.
        description_scale (int): Scale for the description.

    Returns:
        PIL.Image: The inpainted image.
    """
    if mask is None:
        # TODO: Even if there is mask, we shall describe the thing being masked.
        # Perform inference using the Qwen2.5-VL model
        prompt = f"Please describe in detail the {mask_prompt} in the image. Use at least {description_scale} adjective words, including its type, color, texture, etc. Do not use full sentences."
        adjectives = inference(image_path, prompt)

        # Perform masking and inpainting
        inpaint_prompt = f"{adjectives} {inpaint_prompt}"

    output_image = mask_and_inpaint(image_path, mask_prompt, inpaint_prompt, verbose=False, mask=mask)
    
    return output_image

if __name__ == "__main__":
    # Define the input image path and prompts
    input_image_path = "images/beret.jpg"
    mask_prompt = "red hat"
    inpaint_prompt = "blue hat"
    description_scale = 10
    output_image_path = "output_image_refined.jpg"

    # change image path to absolute path
    input_image_path = os.path.abspath(input_image_path)

    output_image = do_inf_inpaint(input_image_path, mask_prompt, inpaint_prompt, description_scale)
    
    output_image.save(output_image_path)

    