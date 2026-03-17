from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.image_utils import Detection, pil_to_rgb_np, rgb_np_to_pil


def thermal_heatmap_from_rgb(image: Image.Image) -> Image.Image:
    """
    Heuristic thermal visualization: grayscale + OpenCV JET colormap.
    """
    rgb = pil_to_rgb_np(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return rgb_np_to_pil(heat)


def overlay_heatmap(base: Image.Image, heatmap_01: np.ndarray, alpha: float = 0.4) -> Image.Image:
    rgb = pil_to_rgb_np(base).astype(np.float32)
    hm = np.clip(heatmap_01, 0.0, 1.0)
    hm_u8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB).astype(np.float32)
    out = (alpha * hm_color + (1.0 - alpha) * rgb).clip(0, 255).astype(np.uint8)
    return rgb_np_to_pil(out)


def draw_detections(image: Image.Image, detections: Iterable[Detection]) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy_int
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=3)
        label = f"{d.class_name} {d.confidence:.2f}"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 3
        draw.rectangle([x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1], fill=(255, 255, 0))
        draw.text((x1 + pad, max(0, y1 - th - pad)), label, fill=(0, 0, 0), font=font)

    return out

