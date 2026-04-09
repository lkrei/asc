import argparse
import json
import time
import sys
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent / "baseline"
PROJECT_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(BASELINE_DIR))
from dataset import ArchitecturalStyleDataset


SUPPORTED_MODELS = {
    "resnet50", "efficientnet_b0", "efficientnet_b3",
    "vit_b_16", "convnext_small", "swin_s",
}


def create_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool, device: str):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        num_ft = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ft, num_classes)
        head_params = list(backbone.fc.parameters())
        backbone_params = [p for n, p in backbone.named_parameters() if "fc" not in n]

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        num_ft = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(num_ft, num_classes)
        head_params = list(backbone.classifier.parameters())
        backbone_params = [p for n, p in backbone.named_parameters() if "classifier" not in n]

    elif model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        num_ft = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(num_ft, num_classes)
        head_params = list(backbone.classifier.parameters())
        backbone_params = [p for n, p in backbone.named_parameters() if "classifier" not in n]

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.vit_b_16(weights=weights)
        num_ft = backbone.heads.head.in_features
        backbone.heads.head = nn.Linear(num_ft, num_classes)
        head_params = list(backbone.heads.parameters())
        backbone_params = [p for n, p in backbone.named_parameters() if "heads" not in n]

    elif model_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_small(weights=weights)
        num_ft = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Linear(num_ft, num_classes)
        head_params = list(backbone.classifier[2].parameters())
        backbone_params = [p for n, p in backbone.named_parameters()
                           if not n.startswith("classifier.2")]

    elif model_name == "swin_s":
        weights = models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.swin_s(weights=weights)
        num_ft = backbone.head.in_features
        backbone.head = nn.Linear(num_ft, num_classes)
        head_params = list(backbone.head.parameters())
        backbone_params = [p for n, p in backbone.named_parameters() if "head" not in n]
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {SUPPORTED_MODELS}")

    if freeze_backbone:
        for p in backbone_params:
            p.requires_grad = False

    backbone = backbone.to(device)
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"Model: {model_name} | trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return backbone, backbone_params, head_params

def get_transforms(mode: str, image_size: int, augmentation: str = "basic"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if mode == "train":
        aug_list = [transforms.Resize(image_size + 32),
                    transforms.RandomCrop(image_size)]

        if augmentation == "autoaugment":
            aug_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        elif augmentation == "randaugment":
            aug_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

        aug_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ]
        return transforms.Compose(aug_list)
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, val_metric: float):
        if self.best_score is None or val_metric > self.best_score + self.min_delta:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, accs = AverageMeter(), AverageMeter()
    pbar = tqdm(loader, desc="  train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(1) == labels).float().mean().item()
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{accs.avg:.4f}")
    return losses.avg, accs.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, accs = AverageMeter(), AverageMeter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = (outputs.argmax(1) == labels).float().mean().item()
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))
    return losses.avg, accs.avg


@torch.no_grad()
def evaluate_test(model, loader, device, class_names):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
    }
    return metrics, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_learning_curves(history, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--augmentation", type=str, default="autoaugment",
                   choices=["basic", "autoaugment", "randaugment"])
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--data-splits", type=str, default=None)
    return p.parse_args()


def main():
    args = build_args()

    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            setattr(args, k.replace("-", "_"), v)

    if args.experiment_name is None:
        mode = "frozen" if args.freeze_backbone else "finetune"
        args.experiment_name = f"{args.model}_{mode}"

    exp_dir = SCRIPT_DIR / "results" / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if args.data_splits:
        splits_file = Path(args.data_splits)
    else:
        splits_file = BASELINE_DIR / "results" / "data_splits.json"

    with open(splits_file) as f:
        splits = json.load(f)

    idx_to_class_file = splits_file.parent / "idx_to_class.json"
    with open(idx_to_class_file) as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[str(i)] for i in range(num_classes)]

    train_tf = get_transforms("train", args.image_size, args.augmentation)
    val_tf = get_transforms("val", args.image_size)

    train_ds = ArchitecturalStyleDataset(samples=splits["train"], transform=train_tf)
    val_ds = ArchitecturalStyleDataset(samples=splits["val"], transform=val_tf)
    test_ds = ArchitecturalStyleDataset(samples=splits["test"], transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model, backbone_params, head_params = create_model(
        args.model, num_classes, pretrained=True,
        freeze_backbone=args.freeze_backbone, device=device,
    )

    if args.freeze_backbone:
        optimizer = optim.Adam(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam([
            {"params": [p for p in backbone_params if p.requires_grad], "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ], weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    early_stop = EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        lr_info = ", ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
        print(f"  train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}  lr=[{lr_info}]")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": v_acc,
                "model_name": args.model,
            }, exp_dir / "checkpoints" / f"best_model.pth")

        early_stop.step(v_acc)
        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed/60:.1f} min | best val_acc={best_val_acc:.4f}")

    with open(exp_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_learning_curves(history, exp_dir / "learning_curves.png")

    best_ckpt = exp_dir / "checkpoints" / "best_model.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    metrics, y_true, y_pred = evaluate_test(model, test_loader, device, class_names)
    metrics["training_time_min"] = round(elapsed / 60, 2)
    metrics["best_val_acc"] = best_val_acc
    metrics["epochs_trained"] = len(history["train_loss"])

    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_confusion_matrix(y_true, y_pred, class_names, exp_dir / "confusion_matrix.png")

    print(f"\nTest Results:")
    print(f"  accuracy:          {metrics['accuracy']:.4f}")
    print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  macro_f1:          {metrics['macro_f1']:.4f}")
    print(f"  weighted_f1:       {metrics['weighted_f1']:.4f}")
    print(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()
