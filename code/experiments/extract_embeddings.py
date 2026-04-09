import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent / "baseline"
sys.path.insert(0, str(BASELINE_DIR))
from dataset import ArchitecturalStyleDataset


def get_backbone_and_hook(model_name: str, checkpoint_path: str, num_classes: int, device: str):
    if model_name == "resnet50":
        backbone = models.resnet50()
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        hook_layer = backbone.avgpool
        embed_dim = 2048
    elif model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0()
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, num_classes)
        hook_layer = backbone.avgpool
        embed_dim = 1280
    elif model_name == "efficientnet_b3":
        backbone = models.efficientnet_b3()
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, num_classes)
        hook_layer = backbone.avgpool
        embed_dim = 1536
    elif model_name == "vit_b_16":
        backbone = models.vit_b_16()
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, num_classes)
        hook_layer = backbone.encoder.ln
        embed_dim = 768
    elif model_name == "convnext_small":
        backbone = models.convnext_small()
        backbone.classifier[2] = nn.Linear(backbone.classifier[2].in_features, num_classes)
        hook_layer = backbone.avgpool
        embed_dim = 768

    elif model_name == "swin_s":
        backbone = models.swin_s()
        backbone.head = nn.Linear(backbone.head.in_features, num_classes)
        hook_layer = backbone.norm
        embed_dim = 768
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = ckpt["model_state_dict"]
        prefix = "backbone."
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        backbone.load_state_dict(state_dict)

    backbone = backbone.to(device)
    backbone.eval()

    embeddings_store = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            feat = output.detach()
            if feat.dim() == 4:
                feat = feat.view(feat.size(0), -1)
            elif feat.dim() == 3:
                feat = feat[:, 0]  # CLS token for ViT
            embeddings_store["feat"] = feat
        else:
            embeddings_store["feat"] = input[0].detach().view(input[0].size(0), -1)

    hook_layer.register_forward_hook(hook_fn)

    return backbone, embeddings_store, embed_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--data-splits", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.data_splits) as f:
        splits = json.load(f)

    idx_to_class_file = Path(args.data_splits).parent / "idx_to_class.json"
    with open(idx_to_class_file) as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)

    val_tf = transforms.Compose([
        transforms.Resize(args.image_size + 32),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    backbone, embed_store, embed_dim = get_backbone_and_hook(
        args.model, args.checkpoint, num_classes, device)
    print(f"Model: {args.model}, embedding dim: {embed_dim}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_embeddings = {}
    all_labels = {}
    all_paths = {}

    for split_name in ["train", "val", "test"]:
        ds = ArchitecturalStyleDataset(samples=splits[split_name], transform=val_tf)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        embeddings = []
        labels = []
        paths = [s["path"] for s in splits[split_name]]

        with torch.no_grad():
            for images, lbls in tqdm(loader, desc=f"  {split_name}"):
                images = images.to(device)
                _ = backbone(images)
                feat = embed_store["feat"].cpu().numpy()
                embeddings.append(feat)
                labels.extend(lbls.numpy())

        all_embeddings[split_name] = np.concatenate(embeddings, axis=0)
        all_labels[split_name] = np.array(labels)
        all_paths[split_name] = paths
        print(f"  {split_name}: {all_embeddings[split_name].shape}")

    np.savez(output_path,
             train_embeddings=all_embeddings["train"],
             train_labels=all_labels["train"],
             val_embeddings=all_embeddings["val"],
             val_labels=all_labels["val"],
             test_embeddings=all_embeddings["test"],
             test_labels=all_labels["test"])

    paths_file = output_path.with_suffix(".paths.json")
    with open(paths_file, "w") as f:
        json.dump(all_paths, f, indent=2)

    print(f"\nEmbeddings saved to {output_path}")
    print(f"Paths saved to {paths_file}")


if __name__ == "__main__":
    main()
