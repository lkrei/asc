import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_embeddings(path, split="test"):
    data = np.load(path)
    emb = data[f"{split}_embeddings"]
    labels = data[f"{split}_labels"]
    return emb, labels


def load_class_names(splits_dir):
    idx_file = Path(splits_dir) / "idx_to_class.json"
    if idx_file.exists():
        with open(idx_file) as f:
            idx_to_class = json.load(f)
        return [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    return None


def run_tsne(embeddings, perplexity=30, seed=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, init="pca", learning_rate="auto")
    return tsne.fit_transform(embeddings)


def plot_single(coords, labels, class_names, title, save_path):
    n_classes = len(set(labels))
    cmap = plt.cm.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(14, 10))
    for i in range(n_classes):
        mask = labels == i
        name = class_names[i] if class_names else str(i)
        short_name = name[:20] + "..." if len(name) > 20 else name
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                   label=short_name, s=8, alpha=0.6)

    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6,
              markerscale=2, frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(coords1, labels1, coords2, labels2, class_names,
                    title1, title2, save_path):
    n_classes = len(set(labels1))
    cmap = plt.cm.get_cmap("tab20", n_classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    for ax, coords, labels, title in [(ax1, coords1, labels1, title1),
                                       (ax2, coords2, labels2, title2)]:
        for i in range(n_classes):
            mask = labels == i
            name = class_names[i] if class_names else str(i)
            short_name = name[:20] + "..." if len(name) > 20 else name
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                       label=short_name, s=8, alpha=0.6)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, lbl = ax2.get_legend_handles_labels()
    fig.legend(handles, lbl, loc="center right", fontsize=6,
               markerscale=2, bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to .npz embeddings (e.g. frozen baseline)")
    parser.add_argument("--embeddings-ft", type=str, default=None,
                        help="Path to fine-tuned .npz embeddings for comparison")
    parser.add_argument("--splits-dir", type=str, default=None,
                        help="Dir with idx_to_class.json")
    parser.add_argument("--output", type=str, default="../../data/visualizations")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--perplexity", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.splits_dir:
        class_names = load_class_names(args.splits_dir)
    else:
        for candidate in [
            Path(__file__).resolve().parent.parent / "baseline" / "results",
            Path(args.embeddings).parent.parent / "splits",
        ]:
            class_names = load_class_names(candidate)
            if class_names:
                break

    print("Loading embeddings...")
    emb1, labels1 = load_embeddings(args.embeddings, args.split)
    print(f"  Embeddings 1: {emb1.shape}")

    print("Running t-SNE (this may take a minute)...")
    coords1 = run_tsne(emb1, perplexity=args.perplexity)

    name1 = Path(args.embeddings).stem
    plot_single(coords1, labels1, class_names, f"t-SNE: {name1}",
                output_dir / f"tsne_{name1}.png")

    if args.embeddings_ft:
        emb2, labels2 = load_embeddings(args.embeddings_ft, args.split)
        print(f"  Embeddings 2: {emb2.shape}")
        coords2 = run_tsne(emb2, perplexity=args.perplexity)

        name2 = Path(args.embeddings_ft).stem
        plot_single(coords2, labels2, class_names, f"t-SNE: {name2}",
                    output_dir / f"tsne_{name2}.png")

        plot_comparison(coords1, labels1, coords2, labels2, class_names,
                        f"Frozen: {name1}", f"Fine-tuned: {name2}",
                        output_dir / "tsne_comparison.png")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
