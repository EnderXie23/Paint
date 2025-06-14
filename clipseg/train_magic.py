import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

# ==== CLIPSeg import & loader helper ====
from models.clipseg import CLIPDensePredT

def clipseg_model(device, advanced=True):
    """Loads CLIPSeg model in inference mode."""
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=advanced)
    model.eval()
    weights = 'rd64-uni-refined.pth' if advanced else 'rd64-uni.pth'
    ckpt = torch.load(f'../clipseg/weights/{weights}', map_location=device)
    model.load_state_dict(ckpt, strict=False)
    return model

# ==== WandB setup ====
wandb.init(project="clipseg-segmentation", name="ViT")
config = {
    "learning_rate": 5e-3,
    "weight_decay": 1e-4,
    "epochs": 10,
    "batch_size": 8,
    "eta_min": 1e-6,
    "use_tqdm": True,
}
wandb.config.update(config)

# ==== Device & model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = clipseg_model(device, advanced=True).to(device)

# (Optionally) freeze the CLIP backbone and only train segmentation head:
for name, p in seg_model.named_parameters():
    if name.startswith("clip"):
        # print(f"Setting param {name} to require grad.")
        p.requires_grad = False

# exit(0)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, seg_model.parameters()),
                       lr=config["learning_rate"],
                       weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=config["epochs"],
                                                 eta_min=config["eta_min"])

# ==== Loss definitions ====
def dice_loss(logits, target, smooth=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2,3))
    union = prob.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(logits, target, alpha=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    d = dice_loss(logits, target)
    return alpha * bce + (1 - alpha) * d

criterion = combined_loss

# ==== Dataset & DataLoader ====
from dataset import MagicBrushDataset

resize = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_ds = MagicBrushDataset(root_dir="/nvme0n1/xmy/MagicBrush",
                             split="train",
                             transform=resize)
test_ds  = MagicBrushDataset(root_dir="/nvme0n1/xmy/MagicBrush",
                             split="test",
                             transform=resize)

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"], shuffle=False)

# ==== Training loop ====
best_iou = 0.0
for epoch in range(config["epochs"]):
    seg_model.train()
    running_loss = 0.0

    if config["use_tqdm"]:
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{config['epochs']}")

    for imgs, masks, prompts in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        # Forward
        logits = seg_model(imgs, list(prompts))[0]
        # probs = torch.sigmoid(logits)
        loss = criterion(logits, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        wandb.log({"train/loss": loss.item(), "epoch": epoch})

        if config["use_tqdm"]:
            pbar.set_postfix(loss=loss.item())
            pbar.update()

    if config["use_tqdm"]:
        pbar.close()

    avg_loss = running_loss / len(train_ds)
    torch.save(seg_model.state_dict(), f"weights/clipseg_epoch{epoch+1}.pth")

    # ==== Validation ====
    seg_model.eval()
    total_iou, n = 0.0, 0
    with torch.no_grad():
        if config["use_tqdm"]:
            vbar = tqdm(test_loader, desc=f"Val {epoch+1}/{config['epochs']}")

        saved_example = False
        for imgs, masks, prompts in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = seg_model(imgs, list(prompts))[0]

            # dynamic threshold
            preds = torch.sigmoid(logits)
            cutoff = preds.min() + 0.5*(preds.max()-preds.min())
            pred = (preds >= cutoff).float()

            # [DEBUG] Save the first mask locally in the batch as image
            if not saved_example:
                # pull out the first item in the batch
                id = 5
                img0  = imgs[id].cpu()           # tensor shape (3, H, W)
                gt0   = masks[id,0].cpu()        # tensor shape (H, W)
                pred0 = pred[id,0].cpu()         # tensor shape (H, W)
                prompt0 = prompts[id]

                # un-normalize / scale back to [0,255] if needed; here if you used ToTensor()
                from torchvision.utils import save_image
                # save the RGB image
                save_image(img0, f"res/clipseg_input_epoch{epoch+1}.png")
                # save ground truth mask
                save_image(gt0.unsqueeze(0), f"res/clipseg_gt_epoch{epoch+1}.png")
                # save predicted mask
                save_image(pred0.unsqueeze(0), f"res/clipseg_pred_epoch{epoch+1}.png")
                # save prompt
                with open("res/prompt.txt", "w") as f:
                    f.write(prompt0)

                saved_example = True

            inter = (pred * masks).sum(dim=[1,2,3])
            union = ((pred + masks) >= 1).float().sum(dim=[1,2,3])
            total_iou += (inter / (union + 1e-6)).sum().item()
            n += imgs.size(0)

            if config["use_tqdm"]:
                vbar.set_postfix(iou=(inter/ (union + 1e-6)).mean().item())
                vbar.update()

        if config["use_tqdm"]:
            vbar.close()

    avg_iou = total_iou / n
    wandb.log({"val/avg_iou": avg_iou, "train/avg_loss": avg_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch})
    scheduler.step()

    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val IoU={avg_iou:.4f}")

    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(seg_model.state_dict(), "clipseg_best.pth")
        print(f"→ New best model saved (IoU={best_iou:.4f})")
