import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent / "baseline"
sys.path.insert(0, str(BASELINE_DIR))


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, output[0].detach().cpu().numpy()


def get_target_layer(model_name: str, backbone: nn.Module):
    if model_name == "resnet50":
        return backbone.layer4[-1].conv3
    elif model_name.startswith("efficientnet"):
        return backbone.features[-1][0]
    elif model_name == "convnext_small":
        return backbone.features[-1][-1].block[4]
    elif model_name == "swin_s":
        return backbone.features[-1][-1].norm2
    else:
        raise ValueError(f"Grad-CAM not supported for {model_name}")


def load_model(model_name: str, checkpoint_path: str, num_classes: int, device: str):
    if model_name == "resnet50":
        backbone = models.resnet50()
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0()
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b3":
        backbone = models.efficientnet_b3()
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, num_classes)
    elif model_name == "convnext_small":
        backbone = models.convnext_small()
        backbone.classifier[2] = nn.Linear(backbone.classifier[2].in_features, num_classes)
    elif model_name == "swin_s":
        backbone = models.swin_s()
        backbone.head = nn.Linear(backbone.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    prefix = "backbone."
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def preprocess_image(image_path: str, image_size: int = 224):
    tf = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return tf(image).unsqueeze(0), image


def visualize_gradcam(image, cam, pred_class, true_class, class_names, save_path,
                      probs=None, top_k=5):
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (image.size[0], image.size[1]), Image.BILINEAR)) / 255.0

    fig, axes = plt.subplots(1, 3 if probs is not None else 2, figsize=(16, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"True: {true_class}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(cam_resized, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Pred: {pred_class}", fontsize=10)
    axes[1].axis("off")

    if probs is not None:
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_names = [class_names[i][:25] for i in top_indices]
        top_probs = [probs[i] for i in top_indices]
        colors = ["green" if class_names[i] == true_class else "salmon" for i in top_indices]
        axes[2].barh(range(top_k), top_probs[::-1], color=colors[::-1])
        axes[2].set_yticks(range(top_k))
        axes[2].set_yticklabels(top_names[::-1], fontsize=8)
        axes[2].set_xlim(0, 1)
        axes[2].set_title("Top-5 Probabilities")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--data-splits", type=str, required=True)
    parser.add_argument("--output", type=str, default="gradcam_results")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of images per category (correct, incorrect)")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    with open(args.data_splits) as f:
        splits = json.load(f)

    idx_to_class_file = Path(args.data_splits).parent / "idx_to_class.json"
    with open(idx_to_class_file) as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[str(i)] for i in range(num_classes)]

    backbone = load_model(args.model, args.checkpoint, num_classes, device)
    target_layer = get_target_layer(args.model, backbone)
    gradcam = GradCAM(backbone, target_layer)

    test_samples = splits["test"]
    np.random.seed(42)
    np.random.shuffle(test_samples)

    correct_dir = output_dir / "correct"
    incorrect_dir = output_dir / "incorrect"
    correct_dir.mkdir(exist_ok=True)
    incorrect_dir.mkdir(exist_ok=True)

    correct_count = 0
    incorrect_count = 0
    max_per_cat = args.num_samples

    summary = {"correct": [], "incorrect": []}

    for sample in test_samples:
        if correct_count >= max_per_cat * num_classes and incorrect_count >= max_per_cat * num_classes:
            break

        img_path = sample["path"]
        true_label = sample["label"]
        true_name = class_names[true_label]

        if not Path(img_path).exists():
            continue

        try:
            input_tensor, orig_image = preprocess_image(img_path, args.image_size)
            input_tensor = input_tensor.to(device)
            cam, pred_label, logits = gradcam.generate(input_tensor)
            probs = np.exp(logits) / np.exp(logits).sum()
            pred_name = class_names[pred_label]

            is_correct = pred_label == true_label
            if is_correct and correct_count < max_per_cat * num_classes:
                fname = f"{true_name[:20]}_{Path(img_path).stem}.png"
                visualize_gradcam(orig_image, cam, pred_name, true_name, class_names,
                                  correct_dir / fname, probs)
                summary["correct"].append({"path": img_path, "true": true_name, "pred": pred_name,
                                           "confidence": float(probs[pred_label])})
                correct_count += 1
            elif not is_correct and incorrect_count < max_per_cat * num_classes:
                fname = f"{true_name[:20]}_pred_{pred_name[:20]}_{Path(img_path).stem}.png"
                visualize_gradcam(orig_image, cam, pred_name, true_name, class_names,
                                  incorrect_dir / fname, probs)
                summary["incorrect"].append({"path": img_path, "true": true_name, "pred": pred_name,
                                             "confidence": float(probs[pred_label])})
                incorrect_count += 1

        except Exception as e:
            print(f"  Error: {img_path}: {e}")

    with open(output_dir / "gradcam_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nGrad-CAM results saved to {output_dir}")
    print(f"  Correct: {correct_count} images")
    print(f"  Incorrect: {incorrect_count} images")


if __name__ == "__main__":
    main()
