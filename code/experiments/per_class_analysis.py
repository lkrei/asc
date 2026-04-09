import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_metrics(results_dir: Path):
    experiments = {}
    for exp_dir in sorted(results_dir.iterdir()):
        metrics_file = exp_dir / "test_metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            experiments[exp_dir.name] = json.load(f)
    return experiments


def extract_class_names(experiments):
    for data in experiments.values():
        report = data.get("classification_report", {})
        return [k for k in report.keys()
                if k not in ("accuracy", "macro avg", "weighted avg")]
    return []


def build_f1_matrix(experiments, class_names):
    """Rows = styles, Columns = models, Values = F1."""
    rows = {}
    for model_name, data in experiments.items():
        f1_list = data.get("f1_per_class", [])
        if len(f1_list) == len(class_names):
            rows[model_name] = f1_list
    return pd.DataFrame(rows, index=class_names)


def find_confusion_pairs(experiments, class_names, top_n=15):
    """Find the most confused style pairs across all models."""
    all_pairs = {}
    for model_name, data in experiments.items():
        report = data.get("classification_report", {})
        for cls_name in class_names:
            cls_data = report.get(cls_name, {})
            support = cls_data.get("support", 0)
            recall = cls_data.get("recall", 1.0)
            errors = support * (1 - recall)
            if errors > 0:
                pair_key = cls_name
                all_pairs[pair_key] = all_pairs.get(pair_key, 0) + errors

    hardest = sorted(all_pairs.items(), key=lambda x: -x[1])[:top_n]
    return hardest


def find_style_difficulty(experiments, class_names):
    """Average F1 across all models for each style (lower = harder)."""
    f1_sums = {c: 0.0 for c in class_names}
    count = 0
    for data in experiments.values():
        f1_list = data.get("f1_per_class", [])
        if len(f1_list) != len(class_names):
            continue
        count += 1
        for i, c in enumerate(class_names):
            f1_sums[c] += f1_list[i]

    if count == 0:
        return {}
    return {c: f1_sums[c] / count for c in class_names}


def plot_f1_heatmap(df, output_path):
    plt.figure(figsize=(max(12, len(df.columns) * 2.5), max(8, len(df) * 0.4)))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                linewidths=0.5, cbar_kws={"label": "F1 Score"})
    plt.title("Per-Class F1 Score Across Models", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Architectural Style")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_difficulty_ranking(difficulty, output_path):
    sorted_items = sorted(difficulty.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#ff6b6b" if v < 0.5 else "#ffd93d" if v < 0.7 else "#6bcb77" for v in values]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Average F1 Across Models")
    ax.set_title("Style Difficulty Ranking (lower = harder)")
    ax.set_xlim(0, 1)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_improvement_from_finetuning(experiments, class_names, output_path):
    """Show which styles benefit most from fine-tuning vs frozen baseline."""
    frozen_f1 = None
    best_ft_f1 = None
    best_ft_name = None
    best_acc = 0

    for name, data in experiments.items():
        f1 = data.get("f1_per_class", [])
        if len(f1) != len(class_names):
            continue
        if "frozen" in name:
            frozen_f1 = np.array(f1)
        else:
            acc = data.get("accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_ft_f1 = np.array(f1)
                best_ft_name = name

    if frozen_f1 is None or best_ft_f1 is None:
        return

    improvement = best_ft_f1 - frozen_f1
    order = np.argsort(improvement)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#ff6b6b" if v < 0 else "#6bcb77" for v in improvement[order]]
    ax.barh(range(len(class_names)), improvement[order], color=colors)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels([class_names[i] for i in order], fontsize=7)
    ax.set_xlabel("F1 Improvement (fine-tuned - frozen)")
    ax.set_title(f"Per-Style Improvement: {best_ft_name} vs frozen baseline")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--shap-file", type=str, default=None)
    parser.add_argument("--baseline-metrics", type=str, default=None,
                        help="Path to frozen baseline test_metrics.json")
    parser.add_argument("--output", type=str, default="../../data/analysis")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiment_metrics(results_dir)

    if args.baseline_metrics and Path(args.baseline_metrics).exists():
        with open(args.baseline_metrics) as f:
            experiments["resnet50_frozen"] = json.load(f)

    if not experiments:
        print("No results found.")
        return

    class_names = extract_class_names(experiments)
    if not class_names:
        print("Could not extract class names.")
        return

    print(f"Models: {list(experiments.keys())}")
    print(f"Classes: {len(class_names)}")

    # F1 heatmap
    f1_df = build_f1_matrix(experiments, class_names)
    if not f1_df.empty:
        plot_f1_heatmap(f1_df, output_dir / "per_class_f1_heatmap.png")
        f1_df.to_csv(output_dir / "per_class_f1.csv")
        print("  Per-class F1 heatmap saved")

    difficulty = find_style_difficulty(experiments, class_names)
    if difficulty:
        plot_difficulty_ranking(difficulty, output_dir / "difficulty_ranking.png")
        with open(output_dir / "difficulty_ranking.json", "w") as f:
            json.dump(sorted(difficulty.items(), key=lambda x: x[1]), f, indent=2, ensure_ascii=False)
        print("  Difficulty ranking saved")

    hardest = find_confusion_pairs(experiments, class_names)
    print("\n  Hardest styles (most total errors across models):")
    for name, errors in hardest:
        print(f"    {name}: ~{errors:.0f} errors")
    with open(output_dir / "hardest_styles.json", "w") as f:
        json.dump(hardest, f, indent=2, ensure_ascii=False)

    plot_improvement_from_finetuning(experiments, class_names,
                                     output_dir / "finetuning_improvement.png")

    if args.shap_file and Path(args.shap_file).exists():
        with open(args.shap_file) as f:
            shap_data = json.load(f)
        shap_by_feature = {item["feature"]: item["importance"] for item in shap_data}
        with open(output_dir / "shap_vs_difficulty.json", "w") as f:
            combined = {
                "difficulty": difficulty,
                "shap_importance": shap_by_feature,
            }
            json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\nAll analysis saved to {output_dir}")

if __name__ == "__main__":
    main()
