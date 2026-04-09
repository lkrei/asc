from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOCAL_CACHE = Path(__file__).resolve().parent / ".hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_LOCAL_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_LOCAL_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_LOCAL_CACHE / "transformers"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_LOCAL_CACHE / "xdg"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


POSITIVE_PROMPTS = [
    "a photo of a building facade",
    "an exterior photo of a building",
    "a street-level photo of a building exterior",
    "an architectural facade exterior",
    "a front view of a building facade",
    "a photo of a historic building facade",
]

NEGATIVE_PROMPTS = [
    "an interior of a building",
    "a church interior",
    "an architectural floor plan",
    "a blueprint or technical drawing",
    "a painting of a building",
    "an illustration of architecture",
    "a 3d render of a building",
    "a city skyline from far away",
    "a close-up of architectural detail",
    "a room inside a building",
]

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


@dataclass
class StyleConfig:
    style_slug: str
    style_name: str
    style_dir: Path
    images_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="downloads")
    parser.add_argument("--output-dir", default="filtered")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--cache-dir", default=".hf_cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-threshold", type=float, default=0.04)
    parser.add_argument("--review-threshold", type=float, default=0.0)
    parser.add_argument("--min-positive-score", type=float, default=0.24)
    parser.add_argument("--include-review-in-organized", action="store_true")
    parser.add_argument("--organize-mode", choices=("none", "copy", "symlink"), default="none")
    parser.add_argument("--limit-per-style", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def slug_to_style_name(slug: str) -> str:
    return slug.replace("_", " ").strip()


def discover_style_dirs(input_dir: Path) -> list[StyleConfig]:
    styles: list[StyleConfig] = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        images_dir = child / "images"
        if images_dir.exists() and images_dir.is_dir():
            styles.append(
                StyleConfig(
                    style_slug=child.name,
                    style_name=slug_to_style_name(child.name),
                    style_dir=child,
                    images_dir=images_dir,
                )
            )
    return styles


def build_style_prompts(style_name: str) -> list[str]:
    return [
        f"a photo of {style_name}",
        f"a photo of {style_name} architecture",
        f"an exterior photo of {style_name}",
        f"a facade in {style_name}",
    ]


def extract_feature_tensor(output: torch.Tensor | object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr_name in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, attr_name, None)
        if isinstance(value, torch.Tensor):
            if attr_name == "last_hidden_state":
                return value[:, 0, :]
            return value
    raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")


def encode_texts(
    model: CLIPModel,
    processor: AutoProcessor,
    texts: list[str],
    device: str,
) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        text_features = extract_feature_tensor(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(text_features, dim=-1)


def encode_image(
    model: CLIPModel,
    processor: AutoProcessor,
    image_path: Path,
    device: str,
) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        image_features = extract_feature_tensor(model.get_image_features(**inputs))
    return torch.nn.functional.normalize(image_features, dim=-1)


def select_best_prompt(
    scores: torch.Tensor,
    prompts: list[str],
) -> tuple[float, str]:
    best_index = int(torch.argmax(scores).item())
    return float(scores[best_index].item()), prompts[best_index]


def decide_label(
    *,
    pos_score: float,
    neg_score: float,
    margin: float,
    keep_threshold: float,
    review_threshold: float,
    min_positive_score: float,
) -> str:
    if pos_score >= min_positive_score and margin >= keep_threshold:
        return "keep"
    if pos_score >= (min_positive_score - 0.03) and margin >= review_threshold:
        return "review"
    if neg_score > pos_score:
        return "reject"
    return "review"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def organize_image(
    source_path: Path,
    destination_path: Path,
    mode: str,
) -> None:
    ensure_parent(destination_path)
    if destination_path.exists() or destination_path.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(source_path, destination_path)
    elif mode == "symlink":
        destination_path.symlink_to(source_path.resolve())


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    if not input_dir.exists():
        print(f"input dir missing: {input_dir}", file=sys.stderr)
        return 2

    style_dirs = discover_style_dirs(input_dir)
    if not style_dirs:
        print(f"no style dirs with images/ under {input_dir}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    model = CLIPModel.from_pretrained(args.model_id, cache_dir=str(cache_dir)).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(cache_dir))
    model.eval()

    positive_features = encode_texts(model, processor, POSITIVE_PROMPTS, device)
    negative_features = encode_texts(model, processor, NEGATIVE_PROMPTS, device)

    global_summary: list[dict] = []

    for style in style_dirs:
        image_paths = sorted(
            path
            for path in style.images_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        )
        if args.limit_per_style > 0:
            image_paths = image_paths[: args.limit_per_style]

        style_output_dir = output_dir / style.style_slug
        style_output_dir.mkdir(parents=True, exist_ok=True)

        style_prompts = build_style_prompts(style.style_name)
        style_features = encode_texts(model, processor, style_prompts, device)

        rows: list[dict] = []
        keep_count = 0
        review_count = 0
        reject_count = 0

        for image_path in image_paths:
            image_features = encode_image(model, processor, image_path, device)
            pos_scores = (image_features @ positive_features.T).squeeze(0).cpu()
            neg_scores = (image_features @ negative_features.T).squeeze(0).cpu()
            style_scores = (image_features @ style_features.T).squeeze(0).cpu()

            pos_score, best_positive_prompt = select_best_prompt(pos_scores, POSITIVE_PROMPTS)
            neg_score, best_negative_prompt = select_best_prompt(neg_scores, NEGATIVE_PROMPTS)
            style_score, best_style_prompt = select_best_prompt(style_scores, style_prompts)
            margin = pos_score - neg_score

            auto_decision = decide_label(
                pos_score=pos_score,
                neg_score=neg_score,
                margin=margin,
                keep_threshold=args.keep_threshold,
                review_threshold=args.review_threshold,
                min_positive_score=args.min_positive_score,
            )

            if auto_decision == "keep":
                keep_count += 1
            elif auto_decision == "review":
                review_count += 1
            else:
                reject_count += 1

            row = {
                "style_slug": style.style_slug,
                "style_name": style.style_name,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "positive_score": round(pos_score, 6),
                "negative_score": round(neg_score, 6),
                "margin": round(margin, 6),
                "style_score": round(style_score, 6),
                "best_positive_prompt": best_positive_prompt,
                "best_negative_prompt": best_negative_prompt,
                "best_style_prompt": best_style_prompt,
                "auto_decision": auto_decision,
                "review_status": "",
                "final_style": "",
                "notes": "",
            }
            rows.append(row)

            if args.organize_mode != "none":
                target_dir = style_output_dir / auto_decision
                destination = target_dir / image_path.name
                if auto_decision == "review" and not args.include_review_in_organized:
                    pass
                else:
                    organize_image(image_path, destination, args.organize_mode)

        csv_path = style_output_dir / "clip_scores.csv"
        json_path = style_output_dir / "clip_scores.json"
        summary_path = style_output_dir / "summary.json"

        write_csv(rows, csv_path)
        write_json(json_path, rows)
        write_json(
            summary_path,
            {
                "style_slug": style.style_slug,
                "style_name": style.style_name,
                "total_images": len(rows),
                "keep_count": keep_count,
                "review_count": review_count,
                "reject_count": reject_count,
                "model_id": args.model_id,
                "cache_dir": str(cache_dir),
                "device": device,
                "keep_threshold": args.keep_threshold,
                "review_threshold": args.review_threshold,
                "min_positive_score": args.min_positive_score,
            },
        )

        global_summary.append(
            {
                "style_slug": style.style_slug,
                "style_name": style.style_name,
                "total_images": len(rows),
                "keep_count": keep_count,
                "review_count": review_count,
                "reject_count": reject_count,
                "csv_path": str(csv_path),
            }
        )

    write_json(output_dir / "global_summary.json", global_summary)
    write_csv(global_summary, output_dir / "global_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
