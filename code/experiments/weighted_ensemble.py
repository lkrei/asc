from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score


def parse_weights(items: list[str]) -> dict[str, float]:
    weights = {}
    for item in items:
        name, value = item.split("=", 1)
        weights[name] = float(value)
    return weights


def load_labels(splits_path: Path) -> np.ndarray:
    with open(splits_path) as f:
        splits = json.load(f)
    return np.array([sample["label"] for sample in splits["test"]], dtype=np.int64)


def load_class_names(splits_path: Path) -> list[str]:
    idx_to_class_path = splits_path.parent / "idx_to_class.json"
    with open(idx_to_class_path) as f:
        idx_to_class = json.load(f)
    return [idx_to_class[str(i)] for i in range(len(idx_to_class))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--data-splits", type=str, required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = parse_weights(args.weights)
    y_true = load_labels(Path(args.data_splits))
    class_names = load_class_names(Path(args.data_splits))

    weighted_probs = None
    used = []
    total_weight = 0.0

    for exp_name, weight in weights.items():
        logits_path = results_dir / exp_name / "test_logits.npy"
        if not logits_path.exists():
            raise FileNotFoundError(
                f"Missing logits for {exp_name}: {logits_path}. "
                "Make sure Kaggle outputs include test_logits.npy."
            )
        logits = np.load(logits_path)
        probs = softmax(logits, axis=1)
        weighted_probs = probs * weight if weighted_probs is None else weighted_probs + probs * weight
        total_weight += weight
        used.append(exp_name)

    weighted_probs /= total_weight
    y_pred = weighted_probs.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        ),
        "method": "weighted_probability_averaging",
        "weights": weights,
        "models": used,
    }

    with open(output_dir / "weighted_ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Weighted ensemble saved to {output_dir / 'weighted_ensemble_metrics.json'}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"balanced_accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
