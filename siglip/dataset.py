from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms

import torch
import numpy as np

class MagicBrushDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to the MagicBrush dataset directory.
            split (str): "train" or "test", indicating which split to load.
            transform (callable, optional): Optional transform to apply to source images and masks.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.sample_dirs = sorted(os.listdir(self.root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.sample_dirs[idx])
        # Paths for source image, mask, and instruction text
        source_path = os.path.join(sample_dir, "source_img.png")
        mask_path   = os.path.join(sample_dir, "mask_img.png")
        instr_path  = os.path.join(sample_dir, "instructions.txt")
        
        # Load source image and mask
        source_img = Image.open(source_path).convert("RGB")
        mask_img   = Image.open(mask_path).convert("L")   # grayscale mask

        # Apply transforms if provided (e.g., augmentation or tensor conversion for source)
        if self.transform:
            source_img = self.transform(source_img)
            mask_img = self.transform(mask_img)
        else:
            source_img = transforms.ToTensor()(source_img)

        mask_tensor = torch.from_numpy((np.array(mask_img) == 0).astype('uint8'))
        # mask_tensor = mask_tensor.unsqueeze(0)  # shape (1, H, W)

        # Read instruction text
        with open(instr_path, 'r') as f:
            instruction = f.read().strip()
        
        return source_img, mask_tensor.float(), instruction


if __name__ == "__main__":
    # Example usage:
    train_dataset = MagicBrushDataset(root_dir="/nvme0n1/xmy/MagicBrush", split="train", 
                                    transform=transforms.ToTensor())
    test_dataset  = MagicBrushDataset(root_dir="/nvme0n1/xmy/MagicBrush", split="test", 
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print("Length of training dataset:", len(train_dataset))
    print("Length of test dataset:", len(test_dataset))

