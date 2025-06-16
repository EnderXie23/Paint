# coding: utf-8
"""Utilities for evaluating text driven mask prediction models on the
Magic Brush dataset.

The evaluator measures two aspects of the predicted masks:

1. **mIoU** – the mean intersection over union with the dataset masks.
2. **Cluster count difference** – the absolute difference between the
   number of connected regions of the ground truth mask and an estimate
   of regions in the predicted mask obtained via k-means clustering.

Instead of asking for pre-generated masks, this module expects a mask
prediction model (e.g. the ``clipseg_model`` defined in
``masking_inpainting.py``).  The evaluator will load each dataset sample,
generate a mask with the model and compute the metrics automatically.

Example
-------
>>> from masking_inpainting import clipseg_model
>>> model = clipseg_model('cuda')
>>> evaluate_dataset(model, 'MagicBrush/train', device='cuda')
"""
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage import measure
import torch
from torchvision import transforms
from tqdm import tqdm


def _load_mask(path: str) -> np.ndarray:
    """Load a mask image and return a binary numpy array."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 0).astype(np.uint8)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute intersection-over-union for two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def kmeans_cluster_count(mask: np.ndarray, max_k: int = 5) -> int:
    """Estimate the number of regions in a mask using k-means.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask where foreground is `1`.
    max_k : int, optional
        Maximum number of clusters to test. The default is 5.

    Returns
    -------
    int
        Estimated number of clusters/regions in the mask.
    """
    coords = np.column_stack(np.nonzero(mask))
    # print(f"pred_mask foreground pixel count: {len(coords)}")
    if len(coords) <= 1:
        return int(len(coords) > 0)

    if len(coords) > 1000:
        # print(f"Too many foreground pixels ({len(coords)}), downsampling for kmeans...")
        idx = np.random.choice(len(coords), 1000, replace=False)
        coords = coords[idx]

    best_k = 1
    best_score = -1.0
    for k in range(2, min(max_k, len(coords)) + 1):
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(coords)
        score = silhouette_score(coords, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def predict_mask(model, image_path: str, prompt: str, device: str,
                 transform: transforms.Compose) -> np.ndarray:
    """Generate a binary mask using ``model`` for ``image_path`` and ``prompt``."""

    image = Image.open(image_path).convert("RGB").resize((512, 512))
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        preds = model(tensor, [prompt])[0]

    cutoff = preds.min() + 0.50 * (preds.max() - preds.min())
    mask = torch.where(preds > cutoff, 1.0, 0.0)
    mask = mask.squeeze(0).cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(np.uint8)


def evaluate_pair(
    pred_mask: np.ndarray, gt_mask: np.ndarray, max_k: int = 5
) -> Tuple[float, int]:
    """Evaluate a single predicted mask against ground truth."""

    score_iou = iou(pred_mask, gt_mask)
    gt_regions = measure.label(gt_mask).max()
    pred_regions = kmeans_cluster_count(pred_mask, max_k=max_k)
    # gt_regions = measure.label(gt_mask, connectivity=2).max()
    # pred_regions = measure.label(pred_mask, connectivity=2).max()
    cluster_diff = abs(pred_regions - gt_regions)
    return score_iou, cluster_diff


def evaluate_dataset(
    model,
    dataset_dir: str,
    device: str = "cpu",
    transform: transforms.Compose | None = None,
    max_k: int = 5,
) -> Tuple[float, float]:
    """Evaluate a mask prediction model on a Magic Brush split.

    Parameters
    ----------
    model : callable
        Mask prediction model. It must accept a tensor batch and list of
        prompts as in ``clipseg_model``.
    dataset_dir : str
        Path to the dataset split (e.g. ``MagicBrush/train``).
    device : str, optional
        Device to run the model on.
    transform : torchvision transform, optional
        Transform applied to input images before feeding into ``model``.  If
        ``None`` a default normalisation used by ClipSeg is applied.
    max_k : int, optional
        Maximum cluster count considered when estimating number of regions.

    Returns
    -------
    tuple(float, float)
        Mean IoU and mean cluster difference for the dataset.
    """

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    sample_dirs = [
        os.path.join(dataset_dir, d)
        for d in sorted(os.listdir(dataset_dir))
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    scores = []
    for sample in tqdm(sample_dirs, desc="Evaluating samples", unit="sample"):
        # print(f"Processing sample: {sample}")
        src_path = os.path.join(sample, "source_img.png")
        mask_path = os.path.join(sample, "mask_img.png")
        instr_path = os.path.join(sample, "instructions.txt")

        if not (os.path.exists(src_path) and os.path.exists(mask_path) and os.path.exists(instr_path)):
            # print(f"Skipping sample (missing file): {sample}")
            continue

        # with open(instr_path, "r", encoding="utf-8") as f:
        #     instruction = f.read().strip()

        # pred_mask = predict_mask(model, src_path, instruction, device, transform)
        # gt_mask = _load_mask(mask_path)
        try:
            # print("Reading instruction...")
            with open(instr_path, "r", encoding="utf-8") as f:
                instruction = f.read().strip()
            # print("Predicting mask...")
            pred_mask = predict_mask(model, src_path, instruction, device, transform)
            # print("Loading GT mask...")
            gt_mask = _load_mask(mask_path)
        except Exception as e:
            print(f"Error in sample {sample}: {e}")
            continue

        if pred_mask.shape != gt_mask.shape:
            pred_mask_img = Image.fromarray((pred_mask > 0).astype(np.uint8) * 255)
            pred_mask_img = pred_mask_img.resize(gt_mask.shape[::-1], Image.NEAREST)
            pred_mask = np.array(pred_mask_img) > 0
            pred_mask = pred_mask.astype(np.uint8)

        scores.append(evaluate_pair(pred_mask, gt_mask, max_k=max_k))

    if not scores:
        return 0.0, 0.0

    miou = float(np.mean([s[0] for s in scores]))
    mean_cluster_diff = float(np.mean([s[1] for s in scores]))
    return miou, mean_cluster_diff


if __name__ == "__main__":
    import argparse
    from masking_inpainting import clipseg_model

    parser = argparse.ArgumentParser(
        description="Evaluate a mask prediction model on the Magic Brush dataset"
    )
    parser.add_argument(
        "dataset_dir",
        help="Path to a Magic Brush split (e.g. MagicBrush/train)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on",
    )
    args = parser.parse_args()

    device = args.device
    model = clipseg_model(device)
    miou, diff = evaluate_dataset(model, args.dataset_dir, device)
    print(f"mIoU: {miou:.4f}")
    print(f"Mean cluster difference: {diff:.4f}")

