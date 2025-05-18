import torch
import torch.nn as nn

class SigLIPSegmentationModel(nn.Module):
    def __init__(self, embed_dim=768, initial_shape=(16, 16)):
        super(SigLIPSegmentationModel, self).__init__()
        self.embed_dim = embed_dim
        self.initial_shape = initial_shape  # (H0, W0) of the feature map after projection
        H0, W0 = initial_shape
        fused_dim = embed_dim * 2  # 1536 for SigLIP2 (image_emb + text_emb)
        
        # Project combined embedding into an initial feature map
        self.fc = nn.Linear(fused_dim, 256 * H0 * W0)  # project to 256 channels feature map
        self.initial_bn = nn.BatchNorm2d(256)
        
        # Decoder: upsampling layers (ConvTranspose2d blocks)
        # Each layer doubles the spatial resolution
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.bn1 = nn.BatchNorm2d(128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 32x32 -> 64x64
        self.bn2 = nn.BatchNorm2d(64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 64x64 -> 128x128
        self.bn3 = nn.BatchNorm2d(32)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)    # 128x128 -> 256x256
        self.bn4 = nn.BatchNorm2d(16)
        # (Add more up layers if output resolution is larger, or fewer if smaller)
        
        # Final 1x1 conv to produce single-channel mask
        self.conv_final = nn.Conv2d(16, 1, kernel_size=1)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, image_emb, text_emb):
        # Expect image_emb and text_emb of shape (batch, embed_dim)
        # Fuse embeddings
        fused = torch.cat([image_emb, text_emb], dim=1)  # shape: (batch, 2*embed_dim)
        
        # Initial projection to feature map
        x = self.fc(fused)                    # (batch, 256*H0*W0)
        x = x.view(x.size(0), 256, *self.initial_shape)  # reshape to (batch, 256, H0, W0)
        x = self.initial_bn(x)
        x = self.relu(x)
        
        # Upsampling decoder with conv transpose blocks
        x = self.up1(x)   # (batch, 128, 32, 32)
        x = self.bn1(x);  x = self.relu(x)
        x = self.up2(x)   # (batch, 64, 64, 64)
        x = self.bn2(x);  x = self.relu(x)
        x = self.up3(x)   # (batch, 32, 128, 128)
        x = self.bn3(x);  x = self.relu(x)
        x = self.up4(x)   # (batch, 16, 256, 256)
        x = self.bn4(x);  x = self.relu(x)
        # (If needed, continue upsampling until reaching desired HxW)
        
        mask_logits = self.conv_final(x)      # (batch, 1, H, W) logits for mask
        # Note: We output raw logits. During training we use a sigmoid + BCELoss or BCEWithLogitsLoss.
        return mask_logits

if __name__ == "__main__":
    # Example usage
    model = SigLIPSegmentationModel()
    image_emb = torch.randn(1, 768)  # Example image embedding
    text_emb = torch.randn(1, 768)   # Example text embedding
    mask_logits = model(image_emb, text_emb)
    print(mask_logits.shape)  # Should be (1, 1, 256, 256) if initial_shape is (16, 16)
