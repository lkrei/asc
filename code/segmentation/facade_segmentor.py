import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


FACADE_CATEGORIES = [
    "wall",      # 0
    "window",  # 1
    "door",    # 2
    "roof",        # 3
    "balcony",     # 4
    "column",   # 5
    "sky",         # 6
    "vegetation",  # 7
    "ground",   # 8
    "other",       # 9
]

FACADE_COLORS = np.array([
    [180, 120,  80],   # wall - warm brown
    [100, 180, 255],   # window - light blue
    [160,  80,  40],   # door - dark brown
    [200,  50,  50],   # roof - red
    [200, 200, 100],   # balcony - yellow-green
    [150, 150, 150],   # column - gray
    [135, 206, 235],   # sky - sky blue
    [ 50, 160,  50],   # vegetation - green
    [128, 128, 128],   # ground - medium gray
    [ 60,  60,  60],   # other - dark gray
], dtype=np.uint8)

ADE20K_TO_FACADE = {}

_WALL_IDS = [0, 1, 25, 48, 79, 84]
_WINDOW_IDS = [8, 63]
_DOOR_IDS = [14, 58]
_ROOF_IDS = [5, 86, 106]
_BALCONY_IDS = [38, 95]
_COLUMN_IDS = [42, 93]
_SKY_IDS = [2]
_VEG_IDS = [4, 9, 17, 66, 72]
_GROUND_IDS = [3, 6, 11, 13, 29, 46, 52, 53, 59]

for _ids, _cat in [(_WALL_IDS, 0), (_WINDOW_IDS, 1), (_DOOR_IDS, 2),
                    (_ROOF_IDS, 3), (_BALCONY_IDS, 4), (_COLUMN_IDS, 5),
                    (_SKY_IDS, 6), (_VEG_IDS, 7), (_GROUND_IDS, 8)]:
    for _id in _ids:
        ADE20K_TO_FACADE[_id] = _cat


class FacadeSegmentor:

    def __init__(self, model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
                 device: Optional[str] = None, cache_dir: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.processor = SegformerImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, cache_dir=cache_dir).to(device)
        self.model.eval()

        self._build_remap_table()

    def _build_remap_table(self):
        self.remap = np.full(151, 9, dtype=np.int64)  # default -> "other"
        for ade_id, facade_id in ADE20K_TO_FACADE.items():
            self.remap[ade_id] = facade_id

    @torch.no_grad()
    def segment(self, image, return_probs: bool = False
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
       
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        orig_size = image.size  # (W, H)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, 150, h, w)

        upsampled = torch.nn.functional.interpolate(
            logits, size=(orig_size[1], orig_size[0]),
            mode="bilinear", align_corners=False
        )

        ade_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W) 0-149
        facade_mask = self.remap[ade_mask]

        probs_out = None
        if return_probs:
            ade_probs = torch.softmax(upsampled.squeeze(0), dim=0).cpu().numpy()  # (150, H, W)
            probs_out = np.zeros((orig_size[1], orig_size[0], len(FACADE_CATEGORIES)), dtype=np.float32)
            for ade_id in range(150):
                facade_id = self.remap[ade_id]
                probs_out[:, :, facade_id] += ade_probs[ade_id]

        return facade_mask, probs_out

    def mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        return FACADE_COLORS[mask]

    def overlay(self, image, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image)

        color_mask = self.mask_to_color(mask)
        blended = (image.astype(float) * (1 - alpha) + color_mask.astype(float) * alpha).astype(np.uint8)
        return blended


def main():
    parser = argparse.ArgumentParser(description="Facade segmentation with SegFormer")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="mask.png")
    parser.add_argument("--overlay", type=str, default=None)
    parser.add_argument("--model", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    seg = FacadeSegmentor(model_name=args.model, device=args.device, cache_dir=args.cache_dir)

    mask, _ = seg.segment(args.image)

    color_mask = seg.mask_to_color(mask)
    Image.fromarray(color_mask).save(args.output)
    print(f"Mask saved to {args.output}")

    if args.overlay:
        overlay_img = seg.overlay(args.image, mask)
        Image.fromarray(overlay_img).save(args.overlay)
        print(f"Overlay saved to {args.overlay}")

    h, w = mask.shape
    total = h * w
    print("\nCategory distribution:")
    for i, name in enumerate(FACADE_CATEGORIES):
        count = np.sum(mask == i)
        print(f"  {name:<12}: {count:>8} px ({100*count/total:5.1f}%)")


if __name__ == "__main__":
    main()
