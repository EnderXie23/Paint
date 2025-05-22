import os
from torchvision import transforms

from masking_inpainting import mask_and_inpaint
from VLinference import inference

def extract_mask(image_path, mask):
    import cv2
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T
    import uuid
    import os

    # === Step 1: 加载原图并与 mask 尺寸对齐 ===
    input_image = Image.open(image_path).convert("RGB").resize((mask.shape[1], mask.shape[0]))
    input_np = np.array(input_image)

    # === Step 2: 膨胀 mask，尽量包含完整目标 ===
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    # === Step 3: 构造 masked 图：只保留掩码区域，其他设为灰色背景 ===
    masked_np = input_np.copy()
    masked_np[dilated_mask == 0] = (127, 127, 127)  # 灰色背景
    masked_img = Image.fromarray(masked_np)

    # === Step 4: 将 masked_img 存为临时文件，供 VL 模型读取 ===
    os.makedirs("tmp", exist_ok=True)
    masked_path = f"tmp/masked_{uuid.uuid4().hex}.jpg"
    masked_img.save(masked_path)
    
    return masked_path

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

    if mask :
        masked_path = extract_mask(image_path, mask)
        prompt = f"Please describe in detail the main object in the img. Use at least {description_scale} adjective words, including its type, color, texture, etc. Do not use full sentences."
        adjectives = inference(masked_path, prompt)

        # Perform masking and inpainting
        inpaint_prompt = f"{adjectives} {inpaint_prompt}"
        os.remove(masked_path) # delete the temporary masked image

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

    