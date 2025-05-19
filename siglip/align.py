import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clipseg.models.clipseg import CLIPDensePredT
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import SigLIPSegmentationModel
from dataset import MagicBrushDataset
from torchvision import transforms
from tqdm import tqdm
import wandb
from PIL import Image

# ==== Setup wandb ====
wandb.init(project="siglip2-segmentation", name="align")
config = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 10,
    "batch_size": 8,
    "eta_min": 1e-5,
    "use_tqdm": True,
}
wandb.config = config

# ==== Load SigLIP2 model and processor ====
from transformers import AutoModel, AutoProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siglip_model = AutoModel.from_pretrained("./siglip2").to(device)
siglip_processor = AutoProcessor.from_pretrained("./siglip2", do_rescale=False)
siglip_model.eval()  # set to evaluation mode (no dropout, etc.)
for param in siglip_model.parameters():
    param.requires_grad = False  # freeze SigLIP2 weights

# ==== Loss function ====
import torch.nn.functional as F
def distillation_loss(student_logits, teacher_logits, target, temperature=2.0, alpha=0.5):
    """
    student_logits: (B, 1, H, W)
    teacher_logits: (B, 1, H, W)
    target: ground truth mask (B, 1, H, W)
    """
    # Resize logits if needed
    T = temperature
    student_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)

    # Distillation loss
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)

    # Optional: combine with regular supervised loss
    bce_loss = F.binary_cross_entropy_with_logits(student_logits, target)

    return alpha * kd_loss + (1 - alpha) * bce_loss

# ==== Initialize our segmentation model and optimizer ====
seg_model = SigLIPSegmentationModel().to(device)
criterion = distillation_loss
optimizer = optim.Adam(seg_model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["eta_min"])

# seg_model.load_state_dict(torch.load("best_segmentation_model.pth"))

teacher_seg = CLIPDensePredT(
    version="ViT-B/16",
    reduce_dim=64,
    complex_trans_conv=True
).to(device)
teacher_seg.eval()
teacher_seg.load_state_dict(torch.load('../clipseg/weights/rd64-uni-refined.pth', map_location=torch.device(device)), strict=False)

# ==== Load dataset ====
resize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = MagicBrushDataset(root_dir="/nvme0n1/xmy/MagicBrush", split="train", transform=resize_transform)
test_dataset  = MagicBrushDataset(root_dir="/nvme0n1/xmy/MagicBrush", split="test", transform=resize_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# ==== Util Functions ====
def save_mask_img_t(model, image, instruction, filename):
    image = image.unsqueeze(0)

    with torch.inference_mode():
        preds = model(image.repeat(1, 1, 1, 1), [instruction])[0]

    # Binary (0 or 1) results
    cutoff = preds.min() + 0.50 * (preds.max() - preds.min())
    mask = torch.where(preds > cutoff, 1.0, 0.0)

    mask = mask.squeeze(0).cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask[0], mode='L')  # Convert to grayscale
    mask_image.save(filename)

def save_mask_img(model, image, instruction, filename):
    inputs = siglip_processor(
        text=[instruction],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)
    siglip_out = siglip_model(**inputs)
    img_emb_batch = siglip_out.image_embeds
    txt_emb_batch = siglip_out.text_embeds
    logits = model(img_emb_batch, txt_emb_batch)
    
    pred_mask_probs = torch.sigmoid(logits)
    cutoff = pred_mask_probs.min() + 0.50 * (pred_mask_probs.max() - pred_mask_probs.min())
    pred_mask = (pred_mask_probs >= cutoff).float()

    pred_mask = pred_mask.squeeze(1).cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(pred_mask[0], mode='L')  # Convert to grayscale
    mask_image.save(filename)

def get_mask_t(images, instructions):
    # Flatten the list of images and instructions
    # images = torch.cat(images, dim=0)
    with torch.inference_mode():
        preds = teacher_seg(images, instructions)[0]

    return preds

    # Binary (0 or 1) results
    cutoff = preds.min() + 0.50 * (preds.max() - preds.min())
    mask = torch.where(preds > cutoff, 1.0, 0.0)

    return mask

def get_mask(images, instructions):
    inputs = siglip_processor(
        text=instructions,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)
    siglip_out = siglip_model(**inputs)
    img_emb_batch = siglip_out.image_embeds
    txt_emb_batch = siglip_out.text_embeds
    logits = seg_model(img_emb_batch, txt_emb_batch)

    return logits
    
    pred_mask_probs = torch.sigmoid(logits)
    cutoff = pred_mask_probs.min() + 0.50 * (pred_mask_probs.max() - pred_mask_probs.min())
    pred_mask = (pred_mask_probs >= cutoff).float()

    return pred_mask

# image = torch.randn(1, 3, 256, 256).clip(0, 1).to(device)
# print("Mask shape:", get_mask([image, image], ['a cat', 'a dog']).shape)
# print("Mask_t shape:", get_mask_t([image, image], ['a cat', 'a dog']).shape)

# exit(0)

# ==== Training loop ====
for epoch in range(config["epochs"]):
    seg_model.train()
    pbar = tqdm(train_loader, disable=not config["use_tqdm"])
    total_loss = 0.0
    for i, (source_imgs, masks, instructions) in enumerate(pbar):
        masks = masks.to(device)
        source_imgs = source_imgs.to(device)
        logits_pred = get_mask(source_imgs, instructions).to(device)
        logits_gt = get_mask_t(source_imgs, instructions).to(device)

        loss = criterion(logits_pred, logits_gt, masks)
        wandb.log({f"epoch{epoch + 1}/train/loss": loss.item(), "step": i})
        if config["use_tqdm"]:
            pbar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_dataset)
    wandb.log({f"epoch{epoch + 1}/train/avg_loss": avg_loss})
    if config["use_tqdm"]:
        pbar.close()

    # Save model checkpoint
    torch.save(seg_model.state_dict(), f"align/segmentation_model_epoch{epoch + 1}.pth")
    
    # Evaluate on test set
    seg_model.eval()
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, disable=not config["use_tqdm"])
        for i, (source_imgs, masks, instructions) in enumerate(pbar):
            source_imgs = source_imgs.to(device)
            masks = masks.to(device)
            logits = get_mask(source_imgs, instructions)
            # Apply sigmoid to get probabilities in [0,1]
            pred_mask_probs = torch.sigmoid(logits)  # (batch, 1, H, W)

            # Binarize the mask (threshold = 0.5)
            cutoff = pred_mask_probs.min() + 0.50 * (pred_mask_probs.max() - pred_mask_probs.min())
            pred_mask = (pred_mask_probs >= cutoff).float()

            # [DEBUG] Save the first mask locally in the batch as image
            if i == 0:
                _pred_mask = pred_mask.squeeze(1)
                _masks = masks.squeeze(1)
                _pred_mask = _pred_mask.cpu().numpy()
                _masks = _masks.cpu().numpy()
                
                _pred_mask = np.clip(_pred_mask, 0, 1) * 255
                _masks = np.clip(_masks, 0, 1) * 255
                _pred_mask = _pred_mask.astype(np.uint8)
                _masks = _masks.astype(np.uint8)
                _pred_mask = Image.fromarray(_pred_mask[0], mode='L')
                _masks = Image.fromarray(_masks[0], mode='L')

                _pred_mask.save(f"align/pred_mask_epoch{epoch+1}.png")
                _masks.save(f"align/gt_mask_epoch{epoch+1}.png")

            # Calculate IoU
            intersection = torch.sum(pred_mask * masks)
            union = torch.sum(pred_mask) + torch.sum(masks) - intersection
            iou = intersection / union if union > 0 else 0.0
            total_iou += iou.item()
            count += 1

            wandb.log({f"epoch{epoch + 1}/test/IoU": iou.item(), "step": i})
            if config["use_tqdm"]:
                pbar.set_postfix(iou=iou.item())
        avg_iou = total_iou / count
        wandb.log({f"epoch{epoch + 1}/test/avg_IoU": avg_iou})
        if config["use_tqdm"]:
            pbar.close()

    # Save the best model based on IoU
    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(seg_model.state_dict(), "align/best_segmentation_model.pth")
        print(f"Best model saved with IoU: {best_iou:.4f}")
    else:
        print(f"Model not improved. Current best IoU: {best_iou:.4f}")

    # Update learning rate
    scheduler.step()
    
wandb.finish()