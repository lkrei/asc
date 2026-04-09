from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def resolve_image_path(path_str: str, local_roots: list[Path]) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    if len(path.parts) >= 2:
        tail = Path(path.parts[-2]) / path.parts[-1]
        for root in local_roots:
            candidate = root / tail
            if candidate.exists():
                return candidate

    raise FileNotFoundError(path_str)


class TTADataset(Dataset):
    def __init__(self, samples, image_size=224, local_roots: list[Path] | None = None):
        self.samples = samples
        self.local_roots = local_roots or []
        self.base_tf = transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.flip_tf = transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
                transforms.functional.hflip,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = resolve_image_path(sample["path"], self.local_roots)
        image = Image.open(image_path).convert("RGB")
        return self.base_tf(image), self.flip_tf(image), sample["label"]


def create_model(model_name: str, num_classes: int):
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "convnext_small":
        model = models.convnext_small()
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "swin_s":
        model = models.swin_s()
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def load_checkpoint(model_name: str, checkpoint_path: Path, num_classes: int, device: str):
    model = create_model(model_name, num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("backbone.") for k in state_dict):
        state_dict = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-splits", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-data-root", action="append", default=[],
                        help="Optional local dataset root(s) for remapping Kaggle paths")
    args = parser.parse_args()

    splits_path = Path(args.data_splits)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_path) as f:
        splits = json.load(f)
    with open(splits_path.parent / "idx_to_class.json") as f:
        idx_to_class = json.load(f)
    class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_checkpoint(args.model, Path(args.checkpoint), len(class_names), device)

    local_roots = [Path(p) for p in args.local_data_root]
    if not local_roots:
        base = Path(__file__).resolve().parent.parent.parent / "data"
        local_roots = [
            base / "Dataset_1" / "architectural-styles-dataset",
            base / "Dataset_2" / "architectural-styles-dataset",
        ]

    ds = TTADataset(splits["test"], image_size=args.image_size, local_roots=local_roots)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x_orig, x_flip, labels in loader:
            x_orig = x_orig.to(device)
            x_flip = x_flip.to(device)
            probs = torch.softmax(model(x_orig), dim=1)
            probs += torch.softmax(model(x_flip), dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "tta_policy": "original_plus_horizontal_flip",
        "model": args.model,
        "checkpoint": str(args.checkpoint),
    }

    with open(output_dir / "tta_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"TTA metrics saved to {output_dir / 'tta_metrics.json'}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"balanced_accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
