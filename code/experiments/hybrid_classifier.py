import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler


class SimpleMLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ProjectedHybridMLP(nn.Module):
    def __init__(self, embed_dim: int, attr_dim: int, num_classes: int,
                 branch_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(embed_dim, branch_dim),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attr_branch = nn.Sequential(
            nn.Linear(attr_dim, branch_dim),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(branch_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, emb, attr):
        img_feat = self.image_branch(emb)
        attr_feat = self.attr_branch(attr)
        return self.classifier(torch.cat([img_feat, attr_feat], dim=1))


class GatedHybridMLP(nn.Module):
    def __init__(self, embed_dim: int, attr_dim: int, num_classes: int,
                 branch_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(embed_dim, branch_dim),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attr_branch = nn.Sequential(
            nn.Linear(attr_dim, branch_dim),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 2, branch_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(branch_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, emb, attr):
        img_feat = self.image_branch(emb)
        attr_feat = self.attr_branch(attr)
        gate = self.gate(torch.cat([img_feat, attr_feat], dim=1))
        fused = gate * img_feat + (1 - gate) * attr_feat
        return self.classifier(fused)


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, val_metric: float):
        if self.best_score is None or val_metric > self.best_score:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def load_and_align(embeddings_path: str, attributes_path: str, splits_path: str):
    """Load embeddings and attributes, align by image path."""
    emb_data = np.load(embeddings_path)
    with open(Path(embeddings_path).with_suffix(".paths.json")) as f:
        emb_paths = json.load(f)

    attr_df = pd.read_csv(attributes_path)
    attr_features = [c for c in attr_df.columns if c not in ("image_path", "class_name", "label")]

    def basename_key(p):
        parts = Path(p).parts
        return parts[-2] + "/" + parts[-1] if len(parts) >= 2 else parts[-1]

    attr_by_path = {}
    for _, row in attr_df.iterrows():
        key = basename_key(row["image_path"])
        attr_by_path[key] = row[attr_features].values.astype(np.float32)

    attr_dim = len(attr_features)

    result = {}
    for split in ["train", "val", "test"]:
        embeddings = emb_data[f"{split}_embeddings"]
        labels = emb_data[f"{split}_labels"]
        paths = emb_paths[split]

        attrs_list = []
        valid_mask = []
        for i, p in enumerate(paths):
            key = basename_key(p)
            if key in attr_by_path:
                attrs_list.append(attr_by_path[key])
                valid_mask.append(True)
            else:
                attrs_list.append(np.zeros(attr_dim, dtype=np.float32))
                valid_mask.append(False)

        attrs = np.array(attrs_list)
        valid = np.array(valid_mask)

        nan_mask = np.isnan(attrs)
        if nan_mask.any():
            col_means = np.nanmean(attrs, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            inds = np.where(nan_mask)
            attrs[inds] = np.take(col_means, inds[1])

        result[split] = {
            "embeddings": embeddings,
            "attributes": attrs,
            "labels": labels,
            "valid": valid,
        }

        n_valid = valid.sum()
        print(f"  {split}: {len(embeddings)} total, {n_valid} with attributes ({100*n_valid/len(embeddings):.0f}%)")

    return result, attr_features, attr_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--attributes", type=str, required=True)
    parser.add_argument("--data-splits", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/hybrid")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["hybrid", "embedding_only", "attributes_only"],
                        help="Which features to use")
    parser.add_argument("--fusion", type=str, default="concat",
                        choices=["concat", "projected_concat", "gated"],
                        help="Fusion strategy for hybrid mode")
    parser.add_argument("--branch-dim", type=int, default=256,
                        help="Hidden projection size for projected/gated fusion")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_splits) as f:
        splits_data = json.load(f)

    idx_to_class_file = Path(args.data_splits).parent / "idx_to_class.json"
    with open(idx_to_class_file) as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[str(i)] for i in range(num_classes)]

    print("Loading data...")
    data, attr_features, attr_dim = load_and_align(args.embeddings, args.attributes, args.data_splits)

    embed_dim = data["train"]["embeddings"].shape[1]
    print(f"Embedding dim: {embed_dim}, Attribute dim: {attr_dim}")

    scaler_emb = StandardScaler()
    scaler_attr = StandardScaler()

    train_emb = scaler_emb.fit_transform(data["train"]["embeddings"])
    val_emb = scaler_emb.transform(data["val"]["embeddings"])
    test_emb = scaler_emb.transform(data["test"]["embeddings"])

    train_attr = scaler_attr.fit_transform(data["train"]["attributes"])
    val_attr = scaler_attr.transform(data["val"]["attributes"])
    test_attr = scaler_attr.transform(data["test"]["attributes"])

    dual_input_hybrid = args.mode == "hybrid" and args.fusion in {"projected_concat", "gated"}

    if args.mode == "hybrid" and not dual_input_hybrid:
        train_X = np.concatenate([train_emb, train_attr], axis=1)
        val_X = np.concatenate([val_emb, val_attr], axis=1)
        test_X = np.concatenate([test_emb, test_attr], axis=1)
        input_dim = embed_dim + attr_dim
    elif args.mode == "hybrid" and dual_input_hybrid:
        train_X = (train_emb, train_attr)
        val_X = (val_emb, val_attr)
        test_X = (test_emb, test_attr)
        input_dim = embed_dim + attr_dim
    elif args.mode == "embedding_only":
        train_X, val_X, test_X = train_emb, val_emb, test_emb
        input_dim = embed_dim
    else:
        train_X, val_X, test_X = train_attr, val_attr, test_attr
        input_dim = attr_dim

    print(f"Mode: {args.mode}, fusion: {args.fusion}, input dim: {input_dim}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    def make_loader(X, y, shuffle=True):
        if isinstance(X, tuple):
            ds = TensorDataset(
                torch.tensor(X[0], dtype=torch.float32),
                torch.tensor(X[1], dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            )
        else:
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_X, data["train"]["labels"], shuffle=True)
    val_loader = make_loader(val_X, data["val"]["labels"], shuffle=False)
    test_loader = make_loader(test_X, data["test"]["labels"], shuffle=False)

    if args.mode == "hybrid" and args.fusion == "projected_concat":
        model = ProjectedHybridMLP(
            embed_dim=embed_dim,
            attr_dim=attr_dim,
            num_classes=num_classes,
            branch_dim=args.branch_dim,
            dropout=args.dropout,
        )
    elif args.mode == "hybrid" and args.fusion == "gated":
        model = GatedHybridMLP(
            embed_dim=embed_dim,
            attr_dim=attr_dim,
            num_classes=num_classes,
            branch_dim=args.branch_dim,
            dropout=args.dropout,
        )
    else:
        model = SimpleMLPHead(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stop = EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, correct, total = 0.0, 0, 0
        for batch in train_loader:
            if dual_input_hybrid:
                emb_batch, attr_batch, y_batch = batch
                emb_batch = emb_batch.to(device)
                attr_batch = attr_batch.to(device)
                y_batch = y_batch.to(device)
                out = model(emb_batch, attr_batch)
                batch_size = emb_batch.size(0)
            else:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                batch_size = X_batch.size(0)
            optimizer.zero_grad()
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            losses += loss.item() * batch_size
            correct += (out.argmax(1) == y_batch).sum().item()
            total += batch_size

        train_loss = losses / total
        train_acc = correct / total

        model.eval()
        v_losses, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if dual_input_hybrid:
                    emb_batch, attr_batch, y_batch = batch
                    emb_batch = emb_batch.to(device)
                    attr_batch = attr_batch.to(device)
                    y_batch = y_batch.to(device)
                    out = model(emb_batch, attr_batch)
                    batch_size = emb_batch.size(0)
                else:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    out = model(X_batch)
                    batch_size = X_batch.size(0)
                loss = criterion(out, y_batch)
                v_losses += loss.item() * batch_size
                v_correct += (out.argmax(1) == y_batch).sum().item()
                v_total += batch_size

        val_loss = v_losses / v_total
        val_acc = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        early_stop.step(val_acc)

        if epoch % 10 == 0 or epoch == 1 or early_stop.should_stop:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if early_stop.should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed/60:.1f} min | best val_acc={best_val_acc:.4f}")

    best_state = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if dual_input_hybrid:
                emb_batch, attr_batch, y_batch = batch
                emb_batch = emb_batch.to(device)
                attr_batch = attr_batch.to(device)
                preds = model(emb_batch, attr_batch).argmax(1).cpu().numpy()
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                preds = model(X_batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "classification_report": classification_report(y_true, y_pred,
                                                       target_names=class_names, output_dict=True),
        "mode": args.mode,
        "fusion": args.fusion,
        "branch_dim": args.branch_dim,
        "embed_dim": embed_dim,
        "attr_dim": attr_dim,
        "input_dim": input_dim,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "training_time_min": round(elapsed / 60, 2),
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\nTest Results ({args.mode}):")
    print(f"  accuracy:          {metrics['accuracy']:.4f}")
    print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  macro_f1:          {metrics['macro_f1']:.4f}")
    print(f"  weighted_f1:       {metrics['weighted_f1']:.4f}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
