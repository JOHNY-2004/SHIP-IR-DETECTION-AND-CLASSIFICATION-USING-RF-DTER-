from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


def pil_to_rgb_np(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def rgb_np_to_pil(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def pil_to_bytes(image: Image.Image, *, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # xyxy in pixel coords
    bbox_xyxy: Tuple[float, float, float, float]

    @property
    def bbox_xyxy_int(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def detections_to_table_rows(detections: Iterable[Detection]) -> list[dict]:
    rows: list[dict] = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        rows.append(
            {
                "class_id": int(d.class_id),
                "class_name": d.class_name,
                "confidence": float(d.confidence),
                "x_min": float(x1),
                "y_min": float(y1),
                "x_max": float(x2),
                "y_max": float(y2),
                "width": float(max(0.0, x2 - x1)),
                "height": float(max(0.0, y2 - y1)),
            }
        )
    return rows

