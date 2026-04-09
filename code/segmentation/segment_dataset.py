import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from facade_segmentor import FacadeSegmentor, FACADE_CATEGORIES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(data_dir: Path):
    """Return list of (class_name, image_path) tuples."""
    items = []
    for class_folder in sorted(data_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        for img_file in sorted(class_folder.iterdir()):
            if img_file.suffix.lower() in IMAGE_EXTENSIONS and not img_file.name.startswith("."):
                items.append((class_folder.name, img_file))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--save-overlays", action="store_true",
                        help="Also save colored overlay PNGs for visual inspection")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    seg = FacadeSegmentor(model_name=args.model, device=args.device, cache_dir=args.cache_dir)
    print(f"Device: {seg.device}")

    items = discover_images(data_dir)
    print(f"Found {len(items)} images in {data_dir}")

    stats = {}
    errors = []
    start = time.time()

    for class_name, img_path in tqdm(items, desc="Segmenting"):
        mask_dir = output_dir / class_name
        mask_dir.mkdir(parents=True, exist_ok=True)

        mask_path = mask_dir / (img_path.stem + ".npy")
        if mask_path.exists():
            continue

        try:
            mask, _ = seg.segment(str(img_path))
            np.save(mask_path, mask.astype(np.uint8))

            if args.save_overlays:
                overlay_dir = output_dir / "_overlays" / class_name
                overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay = seg.overlay(str(img_path), mask, alpha=0.45)
                Image.fromarray(overlay).save(overlay_dir / (img_path.stem + ".jpg"), quality=85)

            if class_name not in stats:
                stats[class_name] = {"count": 0, "category_pixels": {c: 0 for c in FACADE_CATEGORIES}}
            stats[class_name]["count"] += 1
            h, w = mask.shape
            for i, cat in enumerate(FACADE_CATEGORIES):
                stats[class_name]["category_pixels"][cat] += int(np.sum(mask == i))

        except Exception as e:
            errors.append({"image": str(img_path), "error": str(e)})
            print(f"  Error: {img_path.name}: {e}")

    elapsed = time.time() - start
    print(f"\nSegmentation complete in {elapsed/60:.1f} min")
    print(f"  Processed: {sum(s['count'] for s in stats.values())} images")
    print(f"  Errors: {len(errors)}")

    with open(output_dir / "segmentation_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if errors:
        with open(output_dir / "segmentation_errors.json", "w") as f:
            json.dump(errors, f, indent=2)

    print(f"Stats saved to {output_dir / 'segmentation_stats.json'}")


if __name__ == "__main__":
    main()
