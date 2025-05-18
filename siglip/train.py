import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import SigLIPSegmentationModel
from dataset import MagicBrushDataset
from torchvision import transforms
from tqdm import tqdm

# ==== Load SigLIP2 model and processor ====
from transformers import AutoModel, AutoProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siglip_model = AutoModel.from_pretrained("./siglip2").to(device)
siglip_processor = AutoProcessor.from_pretrained("./siglip2", do_rescale=False)
siglip_model.eval()  # set to evaluation mode (no dropout, etc.)
for param in siglip_model.parameters():
    param.requires_grad = False  # freeze SigLIP2 weights

# ==== Initialize our segmentation model and optimizer ====
seg_model = SigLIPSegmentationModel().to(device)
criterion = nn.BCEWithLogitsLoss()  # binary cross-entropy loss on logits
optimizer = optim.Adam(seg_model.parameters(), lr=1e-3)

# ==== Load dataset ====
resize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = MagicBrushDataset(root_dir="/nvme0n1/xmy/MagicBrush", split="train", transform=resize_transform)
test_dataset  = MagicBrushDataset(root_dir="/nvme0n1/xmy/MagicBrush", split="test", transform=resize_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ==== Training loop ====
num_epochs = 10
for epoch in range(num_epochs):
    seg_model.train()
    total_loss = 0.0
    proc_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for source_imgs, masks, instructions in train_loader:
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
        loss = criterion(logits, masks)                   # compute BCE loss
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * source_imgs.size(0)

        proc_bar.update(1)
        proc_bar.set_postfix(loss=loss.item())
        
    proc_bar.close()
    avg_loss = total_loss / len(train_dataset)
    
    # Evaluate on test set
    seg_model.eval()
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        for source_imgs, masks, instructions in test_loader:
            masks = masks.to(device)
            inputs = siglip_processor(images=list(source_imgs), text=list(instructions), return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k,v in inputs.items()}
            siglip_out = siglip_model(**inputs)
            img_emb_batch = siglip_out.image_embeds
            txt_emb_batch = siglip_out.text_embeds
            logits = seg_model(img_emb_batch, txt_emb_batch)
            # Apply sigmoid to get probabilities in [0,1]
            pred_mask_probs = torch.sigmoid(logits)  # (batch, 1, H, W)
            # Binarize the mask (threshold = 0.5)
            pred_mask = (pred_mask_probs >= 0.5).float()
            # Compute IoU for each sample in the batch
            # Intersection = logical AND of prediction and ground truth
            # Union = logical OR of prediction and ground truth
            intersection = (pred_mask * masks).sum(dim=[1,2,3])  # sum over H,W for each batch item
            union = ((pred_mask + masks) >= 1).float().sum(dim=[1,2,3])
            batch_iou = (intersection / (union + 1e-6))  # add a small epsilon for numerical stability
            total_iou += batch_iou.sum().item()
            count += masks.size(0)
    avg_iou = total_iou / count
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val IoU: {avg_iou:.4f}")
