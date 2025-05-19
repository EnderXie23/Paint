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
wandb.init(project="siglip2-segmentation", name="dice loss")
config = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 10,
    "batch_size": 8,
    "eta_min": 1e-5,
    "use_tqdm": False,
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

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def iou_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()

# Combined loss
def combined_loss(pred, target, alpha=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1 - alpha) * dice

# ==== Initialize our segmentation model and optimizer ====
seg_model = SigLIPSegmentationModel().to(device)
# Load model weights if available
# seg_model.load_state_dict(torch.load("best_segmentation_model.pth"))
# criterion = nn.BCEWithLogitsLoss()  # binary cross-entropy loss on logits
criterion = combined_loss  # use the combined loss function
optimizer = optim.Adam(seg_model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["eta_min"])

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

# ==== Training loop ====
num_epochs = config["epochs"]
best_iou = 0.0
for epoch in range(num_epochs):
    seg_model.train()
    total_loss = 0.0
    if config["use_tqdm"]:
        proc_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for i, (source_imgs, masks, instructions) in enumerate(train_loader):
        # Move data to device
        masks = masks.to(device)  # shape (batch, 1, H, W)
        # Prepare inputs for SigLIP2
        # Note: SiglipProcessor can handle a batch of images and texts
        inputs = siglip_processor(images=list(source_imgs), text=list(instructions), return_tensors="pt", padding=True).to(device)
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            siglip_out = siglip_model(**inputs)
            img_emb_batch = siglip_out.image_embeds       # shape (batch, 768)
            txt_emb_batch = siglip_out.text_embeds        # shape (batch, 768)
        # Forward pass through segmentation model
        logits = seg_model(img_emb_batch, txt_emb_batch)  # (batch, 1, H, W) logits
        # pred_mask_probs = torch.sigmoid(logits)
        # loss = criterion(pred_mask_probs, masks)
        loss = criterion(logits, masks)
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * source_imgs.size(0)

        wandb.log({f"epoch{epoch + 1}/train/loss": loss.item(), "step": i})
        if config["use_tqdm"]:
            proc_bar.update(1)
            proc_bar.set_postfix(loss=loss.item())
        
    if config["use_tqdm"]:
        proc_bar.close()
        proc_bar = tqdm(total=len(test_loader), desc=f"Evaluating epoch {epoch+1}/{num_epochs}", unit="batch")
    avg_loss = total_loss / len(train_dataset)
    
    torch.save(seg_model.state_dict(), f"train/segmentation_model_epoch{epoch + 1}.pth")

    # Evaluate on test set
    seg_model.eval()
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        for i, (source_imgs, masks, instructions) in enumerate(test_loader):
            masks = masks.to(device)
            inputs = siglip_processor(images=list(source_imgs), text=list(instructions), return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k,v in inputs.items()}
            siglip_out = siglip_model(**inputs)
            img_emb_batch = siglip_out.image_embeds
            txt_emb_batch = siglip_out.text_embeds
            logits = seg_model(img_emb_batch, txt_emb_batch)
            # Apply sigmoid to get probabilities in [0,1]
            pred_mask_probs = torch.sigmoid(logits)  # (batch, 1, H, W)
            # print(f"pred_mask_probs: {pred_mask_probs}")

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

                _pred_mask.save(f"train/pred_mask_epoch{epoch+1}.png")
                _masks.save(f"train/gt_mask_epoch{epoch+1}.png")

            # Compute IoU for each sample in the batch
            intersection = (pred_mask * masks).sum(dim=[1,2,3])  # sum over H,W for each batch item
            union = ((pred_mask + masks) >= 1).float().sum(dim=[1,2,3])
            batch_iou = (intersection / (union + 1e-6))  # add a small epsilon for numerical stability
            total_iou += batch_iou.sum().item()
            count += masks.size(0)

            if config["use_tqdm"]:
                proc_bar.update(1)
                proc_bar.set_postfix(iou=batch_iou.mean().item())
    if config["use_tqdm"]:
        proc_bar.close()
    avg_iou = total_iou / count
    wandb.log({f"epoch{epoch + 1}/train/avg_loss": avg_loss, f"epoch{epoch + 1}/val/avg_iou": avg_iou, "lr": scheduler.get_last_lr()[0]})
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val IoU: {avg_iou:.4f}", flush=True)

    if avg_iou >= best_iou:
        best_iou = avg_iou
        torch.save(seg_model.state_dict(), f"train/best_segmentation_model.pth")
        print(f"Saved new best model with IoU: {best_iou:.4f}")
