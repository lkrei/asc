"""
Streamlit demo for architectural style classification with:
  1. Style prediction + top-5 probabilities
  2. Semantic segmentation mask overlay
  3. Facade attributes table (structural + color)
  4. Grad-CAM heatmap
  5. SHAP feature importance from tabular classifier
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent
BASELINE_DIR = APP_DIR.parent
CODE_DIR = BASELINE_DIR.parent
SEG_DIR = CODE_DIR / "segmentation"
DATA_DIR = CODE_DIR.parent / "data"

sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(SEG_DIR))

st.set_page_config(
    page_title="Классификация архитектурных стилей",
    page_icon="🏛️",
    layout="wide"
)

idx_to_class_file = BASELINE_DIR / "results" / "idx_to_class.json"
if idx_to_class_file.exists():
    with open(idx_to_class_file) as f:
        _idx = json.load(f)
    CLASS_NAMES = [_idx[str(i)] for i in range(len(_idx))]
else:
    from config import CLASS_NAMES


@st.cache_resource
def load_model(model_name, model_path):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        st.error(f"Unknown model: {model_name}")
        return None, device

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("backbone.")] = v
    model.load_state_dict(cleaned, strict=False)
    model.eval().to(device)
    return model, device


@st.cache_resource
def load_segmentor():
    try:
        from facade_segmentor import FacadeSegmentor
        cache = str(CODE_DIR.parent / ".cache")
        return FacadeSegmentor(device="cpu", cache_dir=cache)
    except Exception as e:
        st.warning(f"Сегментатор недоступен: {e}")
        return None


@st.cache_data
def load_shap_importance():
    shap_files = {
        "XGBoost": DATA_DIR / "tabular_results" / "shap_importance_xgboost.json",
        "LightGBM": DATA_DIR / "tabular_results" / "shap_importance_lightgbm.json",
    }
    result = {}
    for name, path in shap_files.items():
        if path.exists():
            with open(path) as f:
                result[name] = json.load(f)
    return result


class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, inp, out):
        self.activations = out.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x, cls=None):
        self.model.eval()
        out = self.model(x)
        if cls is None:
            cls = out.argmax(1).item()
        self.model.zero_grad()
        out[0, cls].backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def get_gradcam(model, model_name):
    if model_name == "resnet50":
        layer = model.layer4[-1].conv3
    elif model_name.startswith("efficientnet"):
        layer = model.features[-1][0]
    else:
        return None
    return SimpleGradCAM(model, layer)


_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(model, image, device):
    tensor = _preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, tensor


def compute_full_attributes(mask, image):
    """Extract structural and color attributes from segmentation mask."""
    from facade_segmentor import FACADE_CATEGORIES

    total = mask.size
    eps = 1e-8

    ratios = {}
    for i, name in enumerate(FACADE_CATEGORIES):
        ratios[name] = float(np.sum(mask == i)) / total

    wall = ratios.get("wall", 0)
    building = wall + ratios.get("window", 0) + ratios.get("door", 0) + ratios.get("roof", 0) + \
               ratios.get("balcony", 0) + ratios.get("column", 0)

    structural = {
        "wall %": round(ratios.get("wall", 0) * 100, 1),
        "window %": round(ratios.get("window", 0) * 100, 1),
        "door %": round(ratios.get("door", 0) * 100, 1),
        "roof %": round(ratios.get("roof", 0) * 100, 1),
        "balcony %": round(ratios.get("balcony", 0) * 100, 1),
        "column %": round(ratios.get("column", 0) * 100, 1),
        "sky %": round(ratios.get("sky", 0) * 100, 1),
        "vegetation %": round(ratios.get("vegetation", 0) * 100, 1),
        "ground %": round(ratios.get("ground", 0) * 100, 1),
    }

    derived = {
        "glass/wall": round(ratios.get("window", 0) / (wall + eps), 3),
        "roof/wall": round(ratios.get("roof", 0) / (wall + eps), 3),
        "door/wall": round(ratios.get("door", 0) / (wall + eps), 3),
        "veg/building": round(ratios.get("vegetation", 0) / (building + eps), 3),
        "building %": round(building * 100, 1),
    }

    img_np = np.array(image.resize((mask.shape[1], mask.shape[0])))
    hsv = matplotlib.colors.rgb_to_hsv(img_np / 255.0)

    wall_mask = mask == 0
    color = {}
    if wall_mask.sum() > 100:
        color["wall_hue"] = round(float(hsv[wall_mask, 0].mean()), 3)
        color["wall_sat"] = round(float(hsv[wall_mask, 1].mean()), 3)
        color["wall_val"] = round(float(hsv[wall_mask, 2].mean()), 3)

    roof_mask = mask == 3
    if roof_mask.sum() > 100:
        color["roof_hue"] = round(float(hsv[roof_mask, 0].mean()), 3)
        color["roof_sat"] = round(float(hsv[roof_mask, 1].mean()), 3)
        color["roof_val"] = round(float(hsv[roof_mask, 2].mean()), 3)

    return structural, derived, color


def main():
    st.title("🏛️ Классификация архитектурных стилей")
    st.markdown("Загрузите фотографию фасада здания для определения архитектурного стиля "
                "с интерпретируемым анализом")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Настройки")
        base_ckpt = BASELINE_DIR / "results" / "checkpoints"
        exp_ckpt = CODE_DIR / "experiments" / "results"

        available = {}
        for p in sorted(base_ckpt.glob("best_model_*.pth")):
            name = p.stem.replace("best_model_", "")
            available[f"baseline/{name}"] = (name, str(p))
        for exp in sorted(exp_ckpt.iterdir()) if exp_ckpt.exists() else []:
            bp = exp / "checkpoints" / "best_model.pth"
            cfg = exp / "config.json"
            if bp.exists() and cfg.exists():
                with open(cfg) as f:
                    c = json.load(f)
                available[f"exp/{exp.name}"] = (c.get("model", "resnet50"), str(bp))

        if not available:
            st.error("Нет доступных чекпоинтов")
            return

        choice = st.selectbox("Модель", list(available.keys()))
        model_name, model_path = available[choice]

        st.markdown("#### Визуализации")
        show_segmentation = st.checkbox("Семантическая сегментация", value=True)
        show_gradcam = st.checkbox("Grad-CAM тепловая карта", value=True)
        show_shap = st.checkbox("SHAP важность признаков", value=True)

        st.markdown("---")
        with st.expander("Все стили (25)"):
            for i, s in enumerate(CLASS_NAMES, 1):
                st.text(f"{i}. {s}")

    model, device = load_model(model_name, model_path)
    if model is None:
        return

    uploaded = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Загрузите изображение для начала анализа")
        return

    image = Image.open(uploaded).convert("RGB")
    probs, tensor = predict(model, image, device)
    top5 = np.argsort(probs)[::-1][:5]
    pred_idx = top5[0]

    # --- Row 1: Image + Classification ---
    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.subheader("📸 Загруженное изображение")
        st.image(image, use_container_width=True)
    with col_res:
        st.subheader("🎯 Результат классификации")
        st.markdown(f"### {CLASS_NAMES[pred_idx]}")
        st.markdown(f"Уверенность: **{probs[pred_idx]*100:.1f}%**")
        st.progress(float(probs[pred_idx]))
        st.markdown("#### Top-5 предсказаний:")
        for rank, i in enumerate(top5, 1):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"{rank}. **{CLASS_NAMES[i]}**")
                st.progress(float(probs[i]))
            with col_b:
                st.markdown(f"**{probs[i]*100:.1f}%**")

    # --- Row 2: Segmentation + Attributes + Grad-CAM ---
    st.markdown("---")
    st.subheader("🔍 Интерпретируемый анализ")

    vis_sections = []
    if show_segmentation:
        vis_sections.append("seg")
    if show_gradcam and model_name in ("resnet50", "efficientnet_b0"):
        vis_sections.append("cam")

    seg_mask = None

    if vis_sections:
        cols = st.columns(len(vis_sections))
        col_idx = 0

        if "seg" in vis_sections:
            seg = load_segmentor()
            if seg:
                seg_mask, _ = seg.segment(image)
                overlay = seg.overlay(image, seg_mask, alpha=0.45)
                with cols[col_idx]:
                    st.markdown("**Семантическая сегментация фасада**")
                    st.image(overlay, use_container_width=True)
                    from facade_segmentor import FACADE_CATEGORIES, FACADE_COLORS
                    legend_items = []
                    for i, cat in enumerate(FACADE_CATEGORIES):
                        pct = float(np.sum(seg_mask == i)) / seg_mask.size * 100
                        if pct > 0.5:
                            c = FACADE_COLORS[i]
                            legend_items.append(f"<span style='background-color:rgb({c[0]},{c[1]},{c[2]});"
                                                f"padding:2px 8px;color:white;border-radius:3px;'>"
                                                f"{cat}: {pct:.1f}%</span>")
                    st.markdown(" ".join(legend_items), unsafe_allow_html=True)
                col_idx += 1

        if "cam" in vis_sections:
            gradcam = get_gradcam(model, model_name)
            if gradcam:
                cam_input = _preprocess(image).unsqueeze(0).to(device)
                cam = gradcam(cam_input, cls=pred_idx)
                cam_resized = np.array(Image.fromarray(
                    (cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)) / 255.0

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image)
                ax.imshow(cam_resized, cmap="jet", alpha=0.5)
                ax.axis("off")
                ax.set_title(f"Grad-CAM: {CLASS_NAMES[pred_idx]}", fontsize=12)

                with cols[col_idx]:
                    st.markdown("**Grad-CAM: области внимания модели**")
                    st.pyplot(fig)
                plt.close()

    # --- Row 3: Facade Attributes Table ---
    if show_segmentation and seg_mask is not None:
        st.markdown("---")
        attr_col1, attr_col2, attr_col3 = st.columns(3)

        structural, derived, color = compute_full_attributes(seg_mask, image)

        with attr_col1:
            st.markdown("**Структурные пропорции**")
            df_struct = pd.DataFrame(
                [(k, v) for k, v in structural.items() if v > 0.1],
                columns=["Элемент", "Доля (%)"]
            )
            st.dataframe(df_struct, use_container_width=True, hide_index=True)

        with attr_col2:
            st.markdown("**Производные соотношения**")
            df_derived = pd.DataFrame(
                [(k, v) for k, v in derived.items()],
                columns=["Соотношение", "Значение"]
            )
            st.dataframe(df_derived, use_container_width=True, hide_index=True)

        with attr_col3:
            st.markdown("**Цветовые характеристики (HSV)**")
            if color:
                df_color = pd.DataFrame(
                    [(k, v) for k, v in color.items()],
                    columns=["Признак", "Значение"]
                )
                st.dataframe(df_color, use_container_width=True, hide_index=True)
            else:
                st.info("Недостаточно данных для цветового анализа")

    # --- Row 4: SHAP Feature Importance ---
    if show_shap:
        shap_data = load_shap_importance()
        if shap_data:
            st.markdown("---")
            st.subheader("📊 SHAP: важность фасадных признаков")
            st.caption("Какие атрибуты фасада наиболее важны для классификации (по данным табулярного классификатора)")

            shap_cols = st.columns(len(shap_data))
            for idx, (model_label, features) in enumerate(shap_data.items()):
                with shap_cols[idx]:
                    top_n = features[:15]
                    names = [f["feature"] for f in top_n][::-1]
                    values = [f["importance"] for f in top_n][::-1]

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.barh(range(len(names)), values, color="#ff6b6b")
                    ax.set_yticks(range(len(names)))
                    ax.set_yticklabels(names, fontsize=8)
                    ax.set_xlabel("Mean |SHAP value|")
                    ax.set_title(f"SHAP — {model_label}", fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        else:
            st.info("SHAP данные не найдены. Запустите `tabular_classifier.py` для генерации.")


if __name__ == "__main__":
    main()
