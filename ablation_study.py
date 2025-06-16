import os
import argparse
from PIL import Image
import numpy as np
import torch
from typing import Optional, Union
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
import csv
from skimage.metrics import structural_similarity as ssim_metric
import lpips
import matplotlib.pyplot as plt

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

def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = np.array(img1.convert('L'))
    arr2 = np.array(img2.convert('L'))
    return ssim_metric(arr1, arr2, data_range=255)

def compute_lpips_fn(model, img1: Image.Image, img2: Image.Image) -> float:
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    im1 = t(img1).unsqueeze(0).to(next(model.parameters()).device)
    im2 = t(img2).unsqueeze(0).to(next(model.parameters()).device)
    return model(im1, im2).item()

def save_results_csv(out_csv, results):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant', 'PSNR (mean ± std)', 'SSIM (mean ± std)', 'LPIPS (mean ± std)'])
        for variant, data in results.items():
            psnr_m, psnr_s = data['psnr']
            ssim_m, ssim_s = data['ssim']
            lpips_m, lpips_s = data['lpips']
            writer.writerow([variant,
                             f'{psnr_m:.2f} ± {psnr_s:.2f}',
                             f'{ssim_m:.3f} ± {ssim_s:.3f}',
                             f'{lpips_m:.3f} ± {lpips_s:.3f}'])

def plot_images(outdir, images_dict):
    n = len(images_dict)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axs = [axs]
    for ax, (name, img) in zip(axs, images_dict.items()):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(name, fontsize=18)  
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.subplots_adjust(top=0.88, bottom=0.05)  
    plt.savefig(os.path.join(outdir, 'comparison.png'))
    plt.close()

def run_variants(image_path, mask_prompt, inpaint_prompt, scale):
    return {
        'No CLIP': lambda: run_no_clip(image_path, mask_prompt, inpaint_prompt, scale),
        'Old CLIP': lambda: run_old_clip(image_path, mask_prompt, inpaint_prompt, scale),
        'No Qwen': lambda: run_no_qwen(image_path, mask_prompt, inpaint_prompt)
    }

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
        # '/nvme0n1/xmy/stable-diffusion-2-inpainting',
        '/root/autodl-tmp/Paint/models/SD',
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
        # '/nvme0n1/xmy/stable-diffusion-2-inpainting',
        '/root/autodl-tmp/Paint/models/SD',
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
    parser.add_argument('--repeat', type=int, default=3, help='Number of trials per variant')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    lpips_model = lpips.LPIPS(net='vgg').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='vgg')

    baseline = run_baseline(args.image, args.mask_prompt, args.inpaint_prompt, args.scale)
    baseline_path = os.path.join(args.outdir, 'baseline.png')
    baseline.save(baseline_path)
    print(f'Baseline saved to {baseline_path}')
    print('Running variants...')
    variants = run_variants(args.image, args.mask_prompt, args.inpaint_prompt, args.scale)
    results = {}
    images_to_plot = {}
    images_to_plot['Baseline'] = baseline

    for name, func in variants.items():
        psnr_list, ssim_list, lpips_list = [], [], []
        imgs = []
        for i in range(args.repeat):
            print(f'Running {name} trial {i+1}/{args.repeat}')
            out_img = func()
            imgs.append(out_img)

            ref_img = baseline  
            psnr_list.append(psnr(ref_img, out_img))
            ssim_list.append(compute_ssim(ref_img, out_img))
            lpips_list.append(compute_lpips_fn(lpips_model, ref_img, out_img))

        out_path = os.path.join(args.outdir, f'{name.replace(" ", "_").lower()}.png')
        imgs[-1].save(out_path)
        images_to_plot[name] = imgs[-1]

        results[name] = {
            'psnr': (np.mean(psnr_list), np.std(psnr_list)),
            'ssim': (np.mean(ssim_list), np.std(ssim_list)),
            'lpips': (np.mean(lpips_list), np.std(lpips_list))
        }

    save_results_csv(os.path.join(args.outdir, 'metrics.csv'), results)
    plot_images(args.outdir, images_to_plot)

    for name, stats in results.items():
        print(f"{name}: PSNR={stats['psnr'][0]:.2f}±{stats['psnr'][1]:.2f}, "
              f"SSIM={stats['ssim'][0]:.3f}±{stats['ssim'][1]:.3f}, "
              f"LPIPS={stats['lpips'][0]:.3f}±{stats['lpips'][1]:.3f}")

    # baseline = run_baseline(args.image, args.mask_prompt, args.inpaint_prompt, args.scale)
    # baseline_path = os.path.join(args.outdir, 'baseline.png')
    # baseline.save(baseline_path)

    # no_clip = run_no_clip(args.image, args.mask_prompt, args.inpaint_prompt, args.scale)
    # no_clip_path = os.path.join(args.outdir, 'no_clip.png')
    # no_clip.save(no_clip_path)

    # old_clip = run_old_clip(args.image, args.mask_prompt, args.inpaint_prompt, args.scale)
    # old_clip_path = os.path.join(args.outdir, 'old_clip.png')
    # old_clip.save(old_clip_path)

    # no_qwen = run_no_qwen(args.image, args.mask_prompt, args.inpaint_prompt)
    # no_qwen_path = os.path.join(args.outdir, 'no_qwen.png')
    # no_qwen.save(no_qwen_path)

    # print('PSNR comparison to baseline:')
    # print('  No CLIP mask:', psnr(baseline, no_clip))
    # print('  Old CLIP:', psnr(baseline, old_clip))
    # print('  Without Qwen:', psnr(baseline, no_qwen))


if __name__ == '__main__':
    main()
