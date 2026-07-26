import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES

from .eva_vit import SimpleFeaturePyramid

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        hw_shape = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, hw_shape


class DINOv3Attention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj(x)


class DINOv3SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim=384, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1_g = nn.Linear(embed_dim, hidden_dim)
        self.fc1_x = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1_g(x)) * self.fc1_x(x))


class DINOv3MLP(nn.Module):
    def __init__(self, embed_dim=384, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DINOv3Block(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_type='swiglu',
    ):
        super().__init__()
        self.gamma_1 = nn.Parameter(torch.ones(embed_dim))
        self.gamma_2 = nn.Parameter(torch.ones(embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = DINOv3Attention(embed_dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        if mlp_type == 'swiglu':
            self.mlp = DINOv3SwiGLUFFN(embed_dim, mlp_ratio)
        elif mlp_type == 'mlp':
            self.mlp = DINOv3MLP(embed_dim, mlp_ratio)
        else:
            raise ValueError(f'Unsupported DINOv3 mlp_type: {mlp_type}')

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


@BACKBONES.register_module()
class DINOv3ViT(nn.Module):
    def __init__(
        self,
        img_size=(320, 800),
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=False,
        num_register_tokens=4,
        mlp_type='swiglu',
        sim_fpn=None,
        pretrained=None,
        frozen=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.pretrained = pretrained
        self.frozen = frozen

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            DINOv3Block(embed_dim, num_heads, mlp_ratio, qkv_bias, mlp_type)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.adapter = SimpleFeaturePyramid(**sim_fpn) if sim_fpn is not None else None

    def init_weights(self):
        if not self.pretrained:
            return
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(
            'Loaded DINOv3ViT checkpoint from %s; missing=%d unexpected=%d',
            self.pretrained, len(missing), len(unexpected))
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen:
            self.eval()
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x, img_metas=None):
        patch_tokens, hw_shape = self.patch_embed(x)
        batch_size = patch_tokens.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        reg_token = self.reg_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, reg_token, patch_tokens), dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        patch_start = 1 + self.num_register_tokens
        patch_tokens = x[:, patch_start:]
        height, width = hw_shape
        features = patch_tokens.transpose(1, 2).reshape(batch_size, -1, height, width)

        if self.adapter is not None:
            return self.adapter(features)
        return [features]
