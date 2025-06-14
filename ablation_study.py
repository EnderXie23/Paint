import os
import argparse
from PIL import Image
import numpy as np
import torch
from typing import Optional, Union
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline

from masking_inpainting import mask_and_inpaint, clipseg_model, create_mask
from inference_painting import do_inf_inpaint
from VLinference import inference


def psnr(
    img1: Image.Image,
    img2: Image.Image,
    mask: Optional[Union[Image.Image, np.ndarray]] = None
) -> float:
    """
    Compute PSNR between two images, optionally excluding regions defined by mask.
    """
    # Convert images to float32 arrays
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)

    # Determine valid-pixel mask
    if mask is None:
        # all pixels valid
        valid = np.ones_like(arr1, dtype=bool)
    else:
        # build exclusion mask
        if isinstance(mask, Image.Image):
            mask_arr = np.array(mask.convert('L'))
        else:
            mask_arr = np.array(mask)
        excluded = mask_arr.astype(bool)

        # valid = inverse of excluded
        valid = ~excluded
        # if images have channels, broadcast the 2D mask
        if arr1.ndim == 3 and valid.ndim == 2:
            valid = np.repeat(valid[:, :, np.newaxis], arr1.shape[2], axis=2)

    # Count valid pixels
    n_valid = np.count_nonzero(valid)
    if n_valid == 0:
        return float('nan')  # nothing to compare

    # Compute MSE over valid pixels only
    diff = arr1 - arr2
    mse = np.sum((diff ** 2)[valid]) / n_valid
    if mse == 0:
        return float('inf')

    # PSNR formula (assuming pixel values in [0,255])
    return 20 * np.log10(255.0 / np.sqrt(mse))


def run_baseline(image_path: str, mask_prompt: str, inpaint_prompt: str,
                 scale: int) -> Image.Image:
    return do_inf_inpaint(image_path, mask_prompt, inpaint_prompt, scale)


def run_no_clip(image_path: str, mask_prompt: str, inpaint_prompt: str,
                scale: int) -> Image.Image:
    prompt = (
        f"Please describe in detail the {mask_prompt} in the image. Use at least {scale} "
        "adjective words, including its type, color, texture, etc. Do not use full sentences."
    )
    adjectives = inference(image_path, prompt)
    full_prompt = f"{adjectives} {inpaint_prompt}".strip()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        '/nvme0n1/xmy/stable-diffusion-2-inpainting',
        revision='fp16',
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    image = Image.open(image_path).convert('RGB').resize((512, 512))
    mask = Image.new('L', image.size, color=255)
    return pipe(prompt=full_prompt, image=image, mask_image=mask).images[0]


def run_old_clip(image_path: str, mask_prompt: str, inpaint_prompt: str,
                 scale: int) -> Image.Image:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = clipseg_model(device, advanced=False)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image, mask = create_mask(image_path, transform, [mask_prompt], model)
    mask = mask.squeeze(0)
    mask = transforms.ToPILImage()(mask)

    prompt = (
        f"Please describe in detail the {mask_prompt} in the image. Use at least {scale} "
        "adjective words, including its type, color, texture, etc. Do not use full sentences."
    )
    adjectives = inference(image_path, prompt)
    full_prompt = f"{adjectives} {inpaint_prompt}".strip()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        '/nvme0n1/xmy/stable-diffusion-2-inpainting',
        revision='fp16',
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    return pipe(prompt=full_prompt, image=image, mask_image=mask).images[0]


def run_no_qwen(image_path: str, mask_prompt: str, inpaint_prompt: str) -> Image.Image:
    return mask_and_inpaint(image_path, mask_prompt, inpaint_prompt, verbose=False)


def main():
    parser = argparse.ArgumentParser(description='Run ablation study.')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--mask_prompt', required=True, help='Prompt describing object to remove')
    parser.add_argument('--inpaint_prompt', required=True, help='Prompt describing what to add')
    parser.add_argument('--scale', type=int, default=10, help='Description adjective scale')
    parser.add_argument('--outdir', default='ablation_results', help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    baseline = run_baseline(args.image, args.mask_prompt, args.inpaint_prompt, args.scale)
    baseline_path = os.path.join(args.outdir, 'baseline.png')
    baseline.save(baseline_path)

    no_clip = run_no_clip(args.image, args.mask_prompt, args.inpaint_prompt, args.scale)
    no_clip_path = os.path.join(args.outdir, 'no_clip.png')
    no_clip.save(no_clip_path)

    old_clip = run_old_clip(args.image, args.mask_prompt, args.inpaint_prompt, args.scale)
    old_clip_path = os.path.join(args.outdir, 'old_clip.png')
    old_clip.save(old_clip_path)

    no_qwen = run_no_qwen(args.image, args.mask_prompt, args.inpaint_prompt)
    no_qwen_path = os.path.join(args.outdir, 'no_qwen.png')
    no_qwen.save(no_qwen_path)

    print('PSNR comparison to baseline:')
    print('  No CLIP mask:', psnr(baseline, no_clip))
    print('  Old CLIP:', psnr(baseline, old_clip))
    print('  Without Qwen:', psnr(baseline, no_qwen))


if __name__ == '__main__':
    # main()

    pass
