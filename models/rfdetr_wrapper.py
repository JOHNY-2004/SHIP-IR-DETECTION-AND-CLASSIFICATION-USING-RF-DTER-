from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from utils.image_utils import Detection


@dataclass(frozen=True)
class RFDETRConfig:
    checkpoint_path: str
    threshold: float = 0.5
    resolution: int = 640
    num_classes: Optional[int] = None


class RFDETRModel:
    """
    Thin wrapper around `rfdetr` package API used in the training notebook.
    """

    def __init__(self, cfg: RFDETRConfig):
        self.cfg = cfg
        self.model = None
        self.class_names: Optional[list[str]] = None

    def load(self) -> None:
        from rfdetr import RFDETRMedium  # import lazily

        ckpt = Path(self.cfg.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"RFDETR checkpoint not found: {ckpt}")

        kwargs = {"pretrain_weights": str(ckpt), "resolution": self.cfg.resolution}
        if self.cfg.num_classes is not None:
            kwargs["num_classes"] = int(self.cfg.num_classes)
        m = RFDETRMedium(**kwargs)
        m.optimize_for_inference()
        self.model = m

    def set_class_names(self, class_names: list[str]) -> None:
        self.class_names = class_names

    def predict(self, image: Image.Image, threshold: Optional[float] = None) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("RFDETR model not loaded")

        thr = float(self.cfg.threshold if threshold is None else threshold)
        det = self.model.predict(image, threshold=thr)

        class_ids = det.class_id
        confs = det.confidence
        boxes = det.xyxy  # (N,4) float

        out: list[Detection] = []
        for class_id, conf, box in zip(class_ids, confs, boxes):
            cid = int(class_id)
            name = str(cid)
            if self.class_names and 0 <= cid < len(self.class_names):
                name = self.class_names[cid]
            x1, y1, x2, y2 = [float(v) for v in box]
            out.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=float(conf),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return out

    def torch_model(self):
        """
        Return the raw LWDETR nn.Module for Grad-CAM (needed for hooks).
        RFDETRMedium wraps _ModelContext; _ModelContext.model is the LWDETR.
        """
        if self.model is None:
            return None
        try:
            import torch
        except Exception:
            torch = None
        ctx = getattr(self.model, "model", None)
        lwdetr = getattr(ctx, "model", None) if ctx is not None else None
        if lwdetr is None and torch is not None and isinstance(self.model, torch.nn.Module):
            lwdetr = self.model
        if lwdetr is None or (torch is not None and not isinstance(lwdetr, torch.nn.Module)):
            return None
        return lwdetr

