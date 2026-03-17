from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import re

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet generator (CycleGAN-style) matching the `.kiro` spec:
    - 2 downsamples
    - 9 residual blocks
    - 2 upsamples
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        model: list[nn.Module] = []

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class CycleGANConfig:
    input_size: Tuple[int, int] = (256, 256)


class CycleGANDomainAdapter:
    def __init__(self, device: Optional[torch.device] = None, cfg: Optional[CycleGANConfig] = None):
        self.cfg = cfg or CycleGANConfig()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.generator = ResNetGenerator().to(self.device)
        self.generator.eval()
        self.weights_path: Optional[str] = None

    def load_weights(self, weights_path: str) -> None:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"CycleGAN weights not found: {weights_path}")

        ckpt = torch.load(str(p), map_location="cpu")
        # Supports either full checkpoint dict or raw state_dict.
        if isinstance(ckpt, dict) and "G_opt2ir" in ckpt:
            state = ckpt["G_opt2ir"]
        else:
            state = ckpt

        # Auto-detect generator hyperparams from the checkpoint.
        # This avoids shape mismatches when the checkpoint was trained with a different
        # number of residual blocks than our default.
        ngf = 64
        first_conv = state.get("model.1.weight")
        if isinstance(first_conv, torch.Tensor) and first_conv.ndim == 4 and first_conv.shape[1] == 3 and first_conv.shape[2:] == (7, 7):
            ngf = int(first_conv.shape[0])

        block_idxs = set()
        for k in state.keys():
            m = re.match(r"^model\.(\d+)\.block\.", k)
            if m:
                block_idxs.add(int(m.group(1)))
        n_blocks = len(block_idxs) if block_idxs else 9

        output_nc = 3
        # Try to infer output channels from final 7x7 conv (out, ngf, 7, 7).
        for k, v in state.items():
            if (
                isinstance(v, torch.Tensor)
                and v.ndim == 4
                and int(v.shape[1]) == ngf
                and tuple(v.shape[2:]) == (7, 7)
            ):
                output_nc = int(v.shape[0])

        # Rebuild generator to match checkpoint layout if needed.
        self.generator = ResNetGenerator(ngf=ngf, n_blocks=n_blocks, output_nc=output_nc).to(self.device)

        self.generator.load_state_dict(state, strict=True)
        self.generator.to(self.device).eval()
        self.weights_path = str(p)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(self.cfg.input_size, Image.BICUBIC)
        arr = np.array(image).astype(np.float32) / 255.0
        arr = (arr * 2.0) - 1.0  # [-1, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
        return t.to(self.device)

    def _postprocess(self, t: torch.Tensor) -> Image.Image:
        t = t.detach().float().cpu().squeeze(0)  # C,H,W
        t = (t + 1.0) / 2.0
        t = torch.clamp(t, 0.0, 1.0)

        if t.ndim == 3 and t.shape[0] == 1:
            arr = (t[0].numpy() * 255.0).astype(np.uint8)  # H,W
            img = Image.fromarray(arr, mode="L").convert("RGB")
            return img

        arr = t.permute(1, 2, 0).numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @torch.no_grad()
    def generate_synthetic_ir(self, optical_image: Image.Image) -> Image.Image:
        x = self._preprocess(optical_image)
        y = self.generator(x)
        return self._postprocess(y)

