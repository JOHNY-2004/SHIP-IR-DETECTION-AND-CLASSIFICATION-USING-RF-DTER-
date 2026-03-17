from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

from models.cyclegan import CycleGANDomainAdapter
from models.gradcam import GradCAM
from models.rfdetr_wrapper import RFDETRConfig, RFDETRModel
from utils.image_utils import detections_to_table_rows, pil_to_bytes
from utils.viz import draw_detections, overlay_heatmap, thermal_heatmap_from_rgb


APP_TITLE = "IR Ship Detection and Classification (CycleGAN → RF-DETR)"


def load_class_names_from_coco(coco_json_path: str) -> list[str]:
    import json as _json

    d = _json.load(open(coco_json_path, "r", encoding="utf-8"))
    cats = d.get("categories", [])
    cats_sorted = sorted(cats, key=lambda c: int(c.get("id", 0)))

    # Roboflow COCO exports for this dataset include an umbrella class `boats` at id=0.
    # The trained checkpoint reports 15 classes, so we drop that umbrella label.
    if cats_sorted and str(cats_sorted[0].get("name", "")).strip().lower() == "boats" and int(cats_sorted[0].get("id", -1)) == 0:
        cats_sorted = cats_sorted[1:]

    # Return a simple 0..N-1 indexed list matching model class indices.
    return [str(c.get("name", i)) for i, c in enumerate(cats_sorted)]


def infer_num_classes_from_class_names(class_names: list[str]) -> Optional[int]:
    """
    RF-DETR checkpoints are often trained with N foreground classes.
    This dataset's COCO export includes an umbrella 'boats' class (id=0) that
    is typically not used as a foreground label. If present, we drop it.
    """
    if not class_names:
        return None
    return len(class_names)


def sidebar_models_and_config() -> dict:
    st.sidebar.header("Model setup")

    rf_ckpt_default = "checkpoints/checkpoint_best_total.pth"
    rf_ckpt = st.sidebar.text_input("RFDETR checkpoint path (.pth)", value=rf_ckpt_default)

    coco_default = "IR-boats-1/train/_annotations.coco.json"
    coco_path = st.sidebar.text_input("COCO annotations (for class names)", value=coco_default)

    st.sidebar.divider()
    st.sidebar.subheader("CycleGAN (Optical → Synthetic IR)")

    cyclegan_mode = st.sidebar.radio(
        "CycleGAN weights",
        options=["Not loaded (disable Optical pipeline)", "Upload .pth", "Download from URL"],
        index=0,
    )

    cyclegan_uploaded = None
    cyclegan_url = None
    if cyclegan_mode == "Upload .pth":
        cyclegan_uploaded = st.sidebar.file_uploader("Upload CycleGAN weights (.pth)", type=["pth"])
    elif cyclegan_mode == "Download from URL":
        cyclegan_url = st.sidebar.text_input("Direct URL to CycleGAN .pth")

    st.sidebar.divider()
    st.sidebar.header("Inference parameters")
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    resolution = st.sidebar.selectbox("RFDETR resolution", options=[512, 640, 768, 896], index=1)

    return {
        "rf_ckpt": rf_ckpt,
        "coco_path": coco_path,
        "cyclegan_mode": cyclegan_mode,
        "cyclegan_uploaded": cyclegan_uploaded,
        "cyclegan_url": cyclegan_url,
        "threshold": threshold,
        "resolution": resolution,
    }


@st.cache_resource(show_spinner=False)
def get_rfdetr_model(checkpoint_path: str, resolution: int, threshold: float, coco_path: str) -> RFDETRModel:
    class_names: list[str] = []
    if coco_path and Path(coco_path).exists():
        class_names = load_class_names_from_coco(coco_path)
    num_classes = infer_num_classes_from_class_names(class_names)

    cfg = RFDETRConfig(
        checkpoint_path=checkpoint_path,
        resolution=resolution,
        threshold=threshold,
        num_classes=num_classes,
    )
    m = RFDETRModel(cfg)
    m.load()
    if class_names:
        m.set_class_names(class_names)
    return m


def load_or_get_cyclegan(mode: str, uploaded_file, url: Optional[str]) -> Optional[CycleGANDomainAdapter]:
    if mode == "Not loaded (disable Optical pipeline)":
        return None

    adapter = CycleGANDomainAdapter()
    weights_dir = Path("checkpoints") / "cyclegan"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if mode == "Upload .pth":
        if uploaded_file is None:
            return None
        target = weights_dir / uploaded_file.name
        target.write_bytes(uploaded_file.getvalue())
        adapter.load_weights(str(target))
        return adapter

    if mode == "Download from URL":
        if not url:
            return None
        import requests

        target = weights_dir / Path(url).name
        with st.sidebar.status("Downloading CycleGAN weights...", expanded=False):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            target.write_bytes(r.content)
        adapter.load_weights(str(target))
        return adapter

    return None


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    cfg = sidebar_models_and_config()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        input_type = st.radio("Input type", options=["Infrared image", "Optical image (will synthesize IR)"], horizontal=True)
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        demo = st.checkbox("Use a demo image from dataset", value=False)

    image: Optional[Image.Image] = None
    if demo:
        demo_path = "IR-boats-1/test/9_843_jpg.rf.cf4adc68d71fdafc536037e133624633.jpg"
        if input_type.startswith("Optical"):
            demo_path = "Dataset/testA/0.jpg"
        if Path(demo_path).exists():
            image = Image.open(demo_path)
            with col_left:
                st.caption(f"Demo image: `{demo_path}`")
        else:
            with col_left:
                st.warning(f"Demo file not found at `{demo_path}`")
    elif uploaded is not None:
        image = Image.open(uploaded)

    if image is None:
        st.info("Upload an image (or enable demo) to run the pipeline.")
        return

    with col_left:
        st.image(image, caption="Input", use_column_width=True)

    # Load models
    try:
        rfdetr = get_rfdetr_model(cfg["rf_ckpt"], cfg["resolution"], cfg["threshold"], cfg["coco_path"])
    except Exception as e:
        st.error(f"Failed to load RFDETR checkpoint: {e}")
        return

    cyclegan = load_or_get_cyclegan(cfg["cyclegan_mode"], cfg["cyclegan_uploaded"], cfg["cyclegan_url"])

    if input_type.startswith("Optical") and cyclegan is None:
        st.error("Optical pipeline selected, but CycleGAN weights are not loaded.")
        return

    run = st.button("Run detection + visualizations", type="primary")
    if not run:
        return

    with st.status("Running pipeline...", expanded=True) as status:
        status.write("Preparing input…")

        synthetic_ir = image
        if input_type.startswith("Optical") and cyclegan is not None:
            status.write("Generating synthetic IR (CycleGAN)…")
            synthetic_ir = cyclegan.generate_synthetic_ir(image)

        status.write("Generating thermal heatmap…")
        thermal = thermal_heatmap_from_rgb(synthetic_ir)

        status.write("Running RFDETR detection/classification…")
        detections = rfdetr.predict(synthetic_ir, threshold=cfg["threshold"])

        status.write("Drawing boxes and creating results table…")
        annotated = draw_detections(synthetic_ir, detections)
        rows = detections_to_table_rows(detections)
        df = pd.DataFrame(rows)

        status.write("Generating Grad-CAM / saliency overlay…")
        gradcam_overlay = None
        try:
            torch_model = rfdetr.torch_model()
            if torch_model is not None:
                # LWDETR expects NestedTensor; build it directly to avoid inplace-copy grad conflict
                def _prepare(x):
                    import torch
                    from rfdetr.utilities.tensors import NestedTensor
                    h, w = x.shape[2], x.shape[3]
                    mask = torch.zeros((x.shape[0], h, w), dtype=torch.bool, device=x.device)
                    return NestedTensor(x, mask)
                explainer = GradCAM(torch_model, target_layer=None, prepare_input=_prepare)
                hm = explainer.generate(synthetic_ir).heatmap_01
                gradcam_overlay = overlay_heatmap(synthetic_ir, hm, alpha=0.45)
        except Exception:
            gradcam_overlay = None

        status.update(label="Done", state="complete", expanded=False)

    tabs = st.tabs(["Detections", "Synthetic IR", "Thermal heatmap", "Grad-CAM overlay"])

    with tabs[0]:
        st.subheader("Detection + classification")
        st.image(annotated, caption="Detections (bbox + class + confidence)", use_column_width=True)
        try:
            st.dataframe(df, hide_index=True)
        except TypeError:
            st.dataframe(df)

        st.download_button(
            "Download detections (JSON)",
            data=json.dumps({"detections": rows}, indent=2),
            file_name="detections.json",
            mime="application/json",
        )
        st.download_button(
            "Download annotated image (PNG)",
            data=pil_to_bytes(annotated, format="PNG"),
            file_name="detections_annotated.png",
            mime="image/png",
        )

    with tabs[1]:
        st.subheader("Synthetic IR image")
        st.image(synthetic_ir, use_column_width=True)
        st.download_button(
            "Download synthetic IR (PNG)",
            data=pil_to_bytes(synthetic_ir, format="PNG"),
            file_name="synthetic_ir.png",
            mime="image/png",
        )

    with tabs[2]:
        st.subheader("Thermal heatmap")
        st.image(thermal, use_column_width=True)
        st.download_button(
            "Download thermal heatmap (PNG)",
            data=pil_to_bytes(thermal, format="PNG"),
            file_name="thermal_heatmap.png",
            mime="image/png",
        )

    with tabs[3]:
        st.subheader("Grad-CAM overlay")
        if gradcam_overlay is None:
            st.warning(
                "Grad-CAM overlay could not be generated from the RFDETR backend model object. "
                "If you need true Grad-CAM on a specific backbone layer, we can wire the exact layer once we confirm the internal model structure."
            )
        else:
            st.image(gradcam_overlay, use_column_width=True)
            st.download_button(
                "Download Grad-CAM overlay (PNG)",
                data=pil_to_bytes(gradcam_overlay, format="PNG"),
                file_name="gradcam_overlay.png",
                mime="image/png",
            )


if __name__ == "__main__":
    main()

