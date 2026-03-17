from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.image_utils import pil_to_rgb_np


@dataclass
class GradCAMResult:
    heatmap_01: np.ndarray  # HxW float32 in [0,1]


def find_rfdetr_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Find a suitable Grad-CAM target layer in RF-DETR / LWDETR.
    Uses the backbone projector's first scale (highest-resolution) output,
    which produces (B, C, H, W) feature maps.
    """
    try:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return None
        # Handle backbone.0 wrapper (ModuleList style)
        if hasattr(backbone, "0"):
            backbone = getattr(backbone, "0", backbone)
        projector = getattr(backbone, "projector", None)
        if projector is None:
            return None
        stages = getattr(projector, "stages", None)
        if stages is None or len(stages) == 0:
            return None
        return stages[0]
    except Exception:
        return None


class GradCAM:
    """
    Grad-CAM for torch models with spatial feature maps.
    RF-DETR uses backbone.projector.stages[0] (highest-res feature map).
    Falls back to input-gradient saliency if no suitable layer is found.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        prepare_input: Optional[Callable[[torch.Tensor], object]] = None,
    ):
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device).eval()
        self.prepare_input = prepare_input

        if target_layer is None:
            target_layer = find_rfdetr_target_layer(model)
        self.target_layer = target_layer

        self._features: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        if self.target_layer is not None:
            self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(_m, _in, out):
            self._features = out

        def bwd_hook(_m, _gin, gout):
            self._grads = gout[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        rgb = pil_to_rgb_np(image).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device)

    def _run_forward(self, x: torch.Tensor) -> object:
        if self.prepare_input is not None:
            inp = self.prepare_input(x)
        else:
            inp = x
        return self.model(inp)

    def generate(self, image: Image.Image) -> GradCAMResult:
        x = self._preprocess(image)
        x.requires_grad_(True)

        self._features = None
        self._grads = None
        self.model.zero_grad(set_to_none=True)
        out = self._run_forward(x)

        # LWDETR returns dict with pred_logits; backprop from max class score
        score = None
        if isinstance(out, dict):
            for k in ("pred_logits", "logits", "scores"):
                if k in out and torch.is_tensor(out[k]):
                    t = out[k]
                    score = t.max()
                    break
        if score is None and torch.is_tensor(out):
            score = out.max()
        if score is None:
            tensors = []
            if isinstance(out, (list, tuple)):
                tensors = [t for t in out if torch.is_tensor(t)]
            if tensors:
                score = sum(t.max() for t in tensors)
        if score is None:
            score = x.sum()

        score.backward(retain_graph=False)

        # Grad-CAM from backbone projector features
        if self._features is not None and self._grads is not None:
            feats = self._features
            grads = self._grads
            if feats.ndim == 4 and grads.ndim == 4:
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = (weights * feats).sum(dim=1, keepdim=False)
                cam = torch.relu(cam)
                cam = cam[0]
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                hm = cam.detach().cpu().numpy().astype(np.float32)
                h, w = hm.shape
                # Upsample to original image size
                orig_h, orig_w = image.size[1], image.size[0]
                if (h, w) != (orig_h, orig_w):
                    hm_t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
                    hm_up = torch.nn.functional.interpolate(
                        hm_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                    )
                    hm = hm_up[0, 0].numpy().astype(np.float32)
                return GradCAMResult(heatmap_01=hm)

        # Fallback: input gradient saliency
        if x.grad is None:
            return GradCAMResult(heatmap_01=np.zeros((image.size[1], image.size[0]), dtype=np.float32))
        g = x.grad.detach()[0]
        g = torch.abs(g).mean(dim=0)
        g = g - g.min()
        g = g / (g.max() + 1e-8)
        hm = g.cpu().numpy().astype(np.float32)
        return GradCAMResult(heatmap_01=hm)

