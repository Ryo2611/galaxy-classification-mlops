import torch
import torch.nn as nn


class MaskedAutoencoderViT(nn.Module):
    """Lightweight MAE for RGB image reconstruction from masked patches."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.reconstruction_head = nn.Linear(embed_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = images.shape
        if channels != self.in_channels or height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected images with shape (B, {self.in_channels}, {self.image_size}, {self.image_size})"
            )
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(batch_size, self.num_patches, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size = patches.shape[0]
        grid_size = self.image_size // self.patch_size
        images = patches.view(
            batch_size,
            grid_size,
            grid_size,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        images = images.permute(0, 3, 1, 4, 2, 5).contiguous()
        return images.view(batch_size, self.in_channels, self.image_size, self.image_size)

    def random_mask(self, patches: torch.Tensor):
        batch_size, num_patches, _ = patches.shape
        keep_count = max(1, int(num_patches * (1.0 - self.mask_ratio)))
        noise = torch.rand(batch_size, num_patches, device=patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep_count]

        visible = torch.gather(
            patches,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1]),
        )
        mask = torch.ones(batch_size, num_patches, device=patches.device)
        mask[:, :keep_count] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return visible, mask, ids_restore, ids_keep

    def forward(self, images: torch.Tensor):
        patches = self.patchify(images)
        visible_patches, mask, ids_restore, ids_keep = self.random_mask(patches)

        visible_tokens = self.patch_embed(visible_patches)
        visible_pos = torch.gather(
            self.pos_embed.expand(images.size(0), -1, -1),
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1]),
        )
        encoded_visible = self.encoder(visible_tokens + visible_pos)

        mask_count = self.num_patches - encoded_visible.shape[1]
        mask_tokens = self.mask_token.expand(images.size(0), mask_count, -1)
        decoder_tokens = torch.cat([encoded_visible, mask_tokens], dim=1)
        decoder_tokens = torch.gather(
            decoder_tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, decoder_tokens.shape[-1]),
        )
        decoded = self.decoder(decoder_tokens + self.pos_embed)
        pred_patches = self.reconstruction_head(decoded)

        loss = ((pred_patches - patches) ** 2).mean(dim=-1)
        masked_loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return masked_loss, pred_patches, mask


def build_mae_vit(config: dict):
    model_cfg = config["model"]
    return MaskedAutoencoderViT(
        image_size=config["preprocessing"]["image_size"],
        patch_size=model_cfg.get("patch_size", 16),
        embed_dim=model_cfg.get("embed_dim", 256),
        encoder_depth=model_cfg.get("encoder_depth", 4),
        decoder_depth=model_cfg.get("decoder_depth", 2),
        num_heads=model_cfg.get("num_heads", 4),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        mask_ratio=model_cfg.get("mask_ratio", 0.75),
    )
