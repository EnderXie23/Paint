import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import SiglipModel, SiglipProcessor, AutoModel

class SigLIPSegmentationModel(nn.Module):
    def __init__(self, embed_dim=768, initial_shape=(16, 16),  n_heads=8):
        super(SigLIPSegmentationModel, self).__init__()
        self.embed_dim = embed_dim
        self.initial_shape = initial_shape  # (H0, W0) of the feature map after projection
        H0, W0 = initial_shape

        self.fuse_attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads)
        
        # Project combined embedding into an initial feature map
        self.fc = nn.Linear(embed_dim, 256 * H0 * W0)  # project to 256 channels feature map
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

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize attention weights
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.constant_(p, 0)
        
    def forward(self, image_emb, text_emb):
        # Expect image_emb and text_emb of shape (batch, embed_dim)
        # Fuse embeddings
        seq = torch.stack([image_emb, text_emb], dim=0)   # (2, batch, embed_dim)
        attn_out, _ = self.fuse_attn(seq, seq, seq)       # (2, batch, embed_dim)
        fused = attn_out.mean(dim=0)                      # (batch, embed_dim)
        
        # Initial projection to feature map
        x = self.fc(fused)                               # (batch, 256*H0*W0)
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


class SiglipDenseBase(nn.Module):
    def __init__(
        self,
        version: str = "./siglip2",
        reduce_cond: int = None,
        reduce_dim: int = 128,
        prompt: str = 'fixed',
        n_tokens: int = None
    ):
        super().__init__()
        # Load SigLIP vision encoder
        self.siglip = AutoModel.from_pretrained(version)
        self.model = self.siglip.vision_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Freeze SigLIP backbone
        for p in self.siglip.parameters():
            p.requires_grad_(False)

        self.layer_norm = nn.LayerNorm(
            self.model.config.hidden_size,
            eps=self.model.config.layer_norm_eps
        ).to(self.device)

        # Conditional reduction
        if reduce_cond is not None:
            self.reduce_cond = nn.Linear(768, reduce_cond).to(self.device)
            self.reduce_cond.weight.requires_grad_(False)
            self.reduce_cond.bias.requires_grad_(False)
        else:
            self.reduce_cond = None

        film_in = reduce_cond if reduce_cond is not None else 768
        self.film_mul = nn.Linear(film_in, reduce_dim).to(self.device)
        self.film_add = nn.Linear(film_in, reduce_dim).to(self.device)

        self.reduce = nn.Linear(768, reduce_dim)
        self.prompt_list = ["A photo of {}."] if prompt == 'fixed' else []
        self.n_tokens = n_tokens

    def rescaled_pos_emb(self, new_size):
        # Grab the original [1, seq_len, C] parameter and strip the batch dim
        pos_embed = self.model.embeddings.position_embedding.weight  # [seq_len, C]
        cls_tok = pos_embed[:1]                                 # [1, C]
        grid = pos_embed[1:]                                    # [N, C]

        orig_grid = round(math.sqrt(grid.shape[0]))
        target_tokens = orig_grid * orig_grid
        delta = target_tokens - grid.shape[0]

        if delta > 0:
            pad = grid[-1:].repeat(delta, 1)
            grid = torch.cat([grid, pad], dim=0)
        elif delta < 0:
            grid = grid[:target_tokens]

        C = grid.shape[1]
        grid = grid.transpose(0, 1).view(1, C, orig_grid, orig_grid)
        grid = F.interpolate(grid, size=new_size, mode='bicubic', align_corners=False)

        new_H, new_W = new_size
        flat = (
            grid.view(1, C, new_H * new_W)
                .permute(0, 2, 1)
                .reshape(new_H * new_W, C)
        )

        return torch.cat([cls_tok, flat], dim=0)  # [new_seq_len, C]

    def visual_forward(self, x_inp: torch.Tensor, extract_layers=(), skip=False, mask=None):
        x_inp = x_inp.to(self.device)
        with torch.no_grad():
            embeds = self.model.embeddings(x_inp)          # [B, L, C]
            x      = embeds.permute(1, 0, 2)               # [L, B, C]
            # print("Size of x:", x.shape)

            # add positional embeddings
            # expected = self.model.config.image_size ** 2 + 1
            # if x.shape[0] != expected:
            #     H = W = int(math.sqrt(x.shape[0] - 1))
            #     pe = self.rescaled_pos_emb((H, W)).to(x.dtype)   # [L, C]
            #     print("Size of pe1:", pe.shape)
            # else:
            #     # strip off the batch dim from the stored parameter
            #     pass
            pe = self.model.embeddings.position_embedding.weight.to(x.dtype)  # [L, C]
            # print("Size of pe2:", pe.shape)

            # unsqueeze so pe has shape [L, 1, C] and broadcasts across B
            x = x + pe.unsqueeze(1)

            x = self.layer_norm(x)  # [L, B, C]
            activations = []
            affinities = []

            for i, blk in enumerate(self.model.encoder.layers):
                x = blk(x, attention_mask=None)[0]
                if i in extract_layers:
                    activations.append(x)
                if skip and i == max(extract_layers, default=-1):
                    break

            cls_token = x[0]  # [B, C]
            return cls_token, activations, affinities

    def get_cond_vec(self, conditional, batch_size):
        if isinstance(conditional, str):
            proc = SiglipProcessor.from_pretrained(self.siglip.name_or_path)
            inputs = proc.tokenizer([conditional], return_tensors='pt').to(next(self.parameters()).device)
            with torch.no_grad():
                cond = self.siglip.text_model(**inputs).pooler_output
            return cond.repeat(batch_size, 1)
        elif isinstance(conditional, list):
            assert len(conditional) == batch_size
            proc = SiglipProcessor.from_pretrained(self.siglip.name_or_path)
            inputs = proc.tokenizer(conditional, return_tensors='pt', padding=True).to(next(self.parameters()).device)
            with torch.no_grad():
                cond = self.siglip.text_model(**inputs).pooler_output
            return cond
        elif torch.is_tensor(conditional) and conditional.ndim == 2:
            return conditional
        else:
            raise ValueError("Unsupported conditional type")


class SiglipDensePredT(SiglipDenseBase):
    def __init__(
        self,
        version: str = './siglip2',
        extract_layers=(3, 6, 9),
        cond_layer=0,
        reduce_dim=128,
        n_heads=4,
        prompt='fixed',
        extra_blocks=0,
        reduce_cond=None,
        fix_shift=False,
        learn_trans_conv_only=False,
        limit_to_siglip_only=False,
        upsample=False,
        add_calibration=False,
        rev_activations=False,
        trans_conv=None,
        n_tokens=None,
        complex_trans_conv=False
    ):
        super().__init__(version, reduce_cond, reduce_dim, prompt, n_tokens)
        self.extract_layers = extract_layers
        self.cond_layer = cond_layer
        self.rev_activations = rev_activations
        self.token_shape = (
            self.model.config.image_size // self.model.config.patch_size,
            self.model.config.image_size // self.model.config.patch_size
        )

        kernel = trans_conv or self.model.config.patch_size
        if not complex_trans_conv:
            self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, kernel_size=kernel, stride=kernel)
        else:
            self.trans_conv = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, 3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim, reduce_dim//2, kernel//2, stride=kernel//2),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim//2, 1, kernel//2, stride=kernel//2)
            )

        self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in extract_layers])
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads)
            for _ in extract_layers
        ])
        self.extra_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads)
            for _ in range(extra_blocks)
        ])

        if upsample:
            self.upsample_proj = nn.Conv2d(reduce_dim, 1, 1)
        else:
            self.upsample_proj = None

        if learn_trans_conv_only:
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.trans_conv.parameters():
                p.requires_grad_(True)

        # Init weights
        for m in self.modules():
            m = m.to(self.device)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.TransformerEncoderLayer):
                nn.init.xavier_normal_(m.self_attn.in_proj_weight)
                nn.init.constant_(m.self_attn.in_proj_bias, 0)
                nn.init.xavier_normal_(m.self_attn.out_proj.weight)
                nn.init.constant_(m.self_attn.out_proj.bias, 0)
                for p in m.parameters():
                    p.data = p.data.to(self.device)
            
    def forward(self, inp_image, conditional=None, return_features=False, mask=None):
        inp_image = inp_image.to(self.device)
        bs = inp_image.shape[0]
        cond = self.get_cond_vec(conditional, bs)

        cls_vec, acts, _ = self.visual_forward(
            inp_image,
            extract_layers=[0] + list(self.extract_layers),
            skip=False
        )
        feats = acts[1:]
        feats = feats[::-1] if not self.rev_activations else feats

        a = None
        for i, (activation, blk, reduce_lin) in enumerate(zip(feats, self.blocks, self.reduces)):
            red = reduce_lin(activation)  # [L, B, D]
            a = red if a is None else a + red
            if i == self.cond_layer:
                if self.reduce_cond:
                    cond = self.reduce_cond(cond)
                # broadcast FiLM across token dimension
                film_mul = self.film_mul(cond).unsqueeze(0)  # [1,B,D]
                film_add = self.film_add(cond).unsqueeze(0)  # [1,B,D]
                a = film_mul * a + film_add
            a = blk(a)

        for blk in self.extra_blocks:
            a = a + blk(a)

        _, B, D = a.shape
        size = int(math.sqrt(a.shape[0]))
        a = a.permute(1, 2, 0).view(bs, D, size, size)

        mask_logits = self.trans_conv(a)
        if self.n_tokens:
            mask_logits = F.interpolate(
                mask_logits,
                inp_image.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        if self.upsample_proj:
            mask_logits = self.upsample_proj(mask_logits)
            mask_logits = F.interpolate(
                mask_logits,
                inp_image.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        if return_features:
            return mask_logits, cls_vec, cond, acts
        return mask_logits


if __name__ == "__main__":
    model = SiglipDensePredT()
    image = torch.randn(5, 3, 224, 224)
    pred = model(image, conditional=["A cat", "A dog", "A dog", "A dog", "A dog"])
    print(pred.shape)  # should be [5, 1, 224, 224]