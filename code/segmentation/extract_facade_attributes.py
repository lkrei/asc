import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

from facade_segmentor import FACADE_CATEGORIES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Category indices
CAT_WALL = 0
CAT_WINDOW = 1
CAT_DOOR = 2
CAT_ROOF = 3
CAT_BALCONY = 4
CAT_COLUMN = 5
CAT_SKY = 6
CAT_VEGETATION = 7
CAT_GROUND = 8
CAT_OTHER = 9


def pixel_ratios(mask: np.ndarray) -> dict:
    total = mask.size
    ratios = {}
    for i, name in enumerate(FACADE_CATEGORIES):
        ratios[f"{name}_ratio"] = float(np.sum(mask == i)) / total
    return ratios


def derived_ratios(ratios: dict) -> dict:
    wall = ratios.get("wall_ratio", 0)
    building = wall + ratios.get("roof_ratio", 0) + ratios.get("window_ratio", 0) + \
               ratios.get("door_ratio", 0) + ratios.get("balcony_ratio", 0) + \
               ratios.get("column_ratio", 0)

    eps = 1e-8
    return {
        "glass_to_wall": ratios.get("window_ratio", 0) / (wall + eps),
        "roof_to_wall": ratios.get("roof_ratio", 0) / (wall + eps),
        "door_to_wall": ratios.get("door_ratio", 0) / (wall + eps),
        "vegetation_to_building": ratios.get("vegetation_ratio", 0) / (building + eps),
        "sky_to_total": ratios.get("sky_ratio", 0),
        "building_ratio": building,
    }


def window_geometry(mask: np.ndarray) -> dict:
    window_mask = (mask == CAT_WINDOW).astype(np.uint8)
    if window_mask.sum() == 0:
        return {"window_count": 0, "avg_window_area": 0, "window_regularity": 0}

    labeled, n_components = ndimage.label(window_mask)
    areas = ndimage.sum(window_mask, labeled, range(1, n_components + 1))
    min_area = mask.size * 0.0005
    valid_areas = [a for a in areas if a >= min_area]

    if not valid_areas:
        return {"window_count": 0, "avg_window_area": 0, "window_regularity": 0}

    avg_area = float(np.mean(valid_areas))
    std_area = float(np.std(valid_areas)) if len(valid_areas) > 1 else 0
    regularity = 1.0 - min(std_area / (avg_area + 1e-8), 1.0)

    return {
        "window_count": len(valid_areas),
        "avg_window_area": avg_area / mask.size,
        "window_regularity": regularity,
    }


def symmetry_score(mask: np.ndarray) -> dict:
    h, w = mask.shape

    left = mask[:, :w // 2]
    right = np.flip(mask[:, (w - w // 2):], axis=1)
    min_w = min(left.shape[1], right.shape[1])
    h_sym = float(np.mean(left[:, :min_w] == right[:, :min_w]))

    top = mask[:h // 2, :]
    bottom = np.flip(mask[(h - h // 2):, :], axis=0)
    min_h = min(top.shape[0], bottom.shape[0])
    v_sym = float(np.mean(top[:min_h, :] == bottom[:min_h, :]))

    return {"horizontal_symmetry": h_sym, "vertical_symmetry": v_sym}


def color_features(image: np.ndarray, mask: np.ndarray) -> dict:
    from colorsys import rgb_to_hsv

    features = {}

    for cat_name, cat_id in [("wall", CAT_WALL), ("roof", CAT_ROOF)]:
        region_mask = mask == cat_id
        if region_mask.sum() < 100:
            for suffix in ["_mean_hue", "_mean_sat", "_mean_val", "_std_hue"]:
                features[f"{cat_name}{suffix}"] = 0.0
            continue

        pixels = image[region_mask].astype(np.float32) / 255.0
        hsv = np.array([rgb_to_hsv(r, g, b) for r, g, b in pixels])
        features[f"{cat_name}_mean_hue"] = float(np.mean(hsv[:, 0]))
        features[f"{cat_name}_mean_sat"] = float(np.mean(hsv[:, 1]))
        features[f"{cat_name}_mean_val"] = float(np.mean(hsv[:, 2]))
        features[f"{cat_name}_std_hue"] = float(np.std(hsv[:, 0]))

    return features


def color_diversity(image: np.ndarray, mask: np.ndarray, n_bins: int = 16) -> dict:
    building_mask = np.isin(mask, [CAT_WALL, CAT_WINDOW, CAT_DOOR, CAT_ROOF,
                                    CAT_BALCONY, CAT_COLUMN])
    if building_mask.sum() < 100:
        return {"color_entropy": 0.0}

    pixels = image[building_mask]
    quantized = (pixels // (256 // n_bins)).astype(np.int32)
    codes = quantized[:, 0] * n_bins * n_bins + quantized[:, 1] * n_bins + quantized[:, 2]
    _, counts = np.unique(codes, return_counts=True)
    probs = counts / counts.sum()
    entropy = -float(np.sum(probs * np.log2(probs + 1e-10)))
    max_entropy = np.log2(n_bins ** 3)
    return {"color_entropy": entropy / max_entropy}


def extract_all_features(image_path: str, mask: np.ndarray) -> dict:
    image = np.array(Image.open(image_path).convert("RGB"))

    if image.shape[:2] != mask.shape:
        from PIL import Image as PILImage
        img_resized = PILImage.fromarray(image).resize((mask.shape[1], mask.shape[0]))
        image = np.array(img_resized)

    features = {}
    features.update(pixel_ratios(mask))
    features.update(derived_ratios(features))
    features.update(window_geometry(mask))
    features.update(symmetry_score(mask))

    n_pixels = mask.size
    if n_pixels > 500_000:
        step = max(1, int(np.sqrt(n_pixels / 100_000)))
        image_ds = image[::step, ::step]
        mask_ds = mask[::step, ::step]
    else:
        image_ds, mask_ds = image, mask

    features.update(color_features(image_ds, mask_ds))
    features.update(color_diversity(image_ds, mask_ds))

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="facade_attributes.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    masks_dir = Path(args.masks_dir)
    output_path = Path(args.output)

    items = []
    for class_folder in sorted(data_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        for img_file in sorted(class_folder.iterdir()):
            if img_file.suffix.lower() in IMAGE_EXTENSIONS and not img_file.name.startswith("."):
                mask_file = masks_dir / class_folder.name / (img_file.stem + ".npy")
                if mask_file.exists():
                    items.append((class_folder.name, img_file, mask_file))

    print(f"Found {len(items)} image-mask pairs")

    all_rows = []
    for class_name, img_path, mask_path in tqdm(items, desc="Extracting attributes"):
        try:
            mask = np.load(mask_path)
            features = extract_all_features(str(img_path), mask)
            features["image_path"] = str(img_path)
            features["class_name"] = class_name
            all_rows.append(features)
        except Exception as e:
            print(f"  Error {img_path.name}: {e}")

    if not all_rows:
        print("No features extracted!")
        return

    fieldnames = ["image_path", "class_name"] + [k for k in all_rows[0] if k not in ("image_path", "class_name")]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {output_path}")
    print(f"Features per image: {len(fieldnames) - 2}")


if __name__ == "__main__":
    main()
