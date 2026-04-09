
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

FEATURE_COLUMNS = None  

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    label_enc = LabelEncoder()
    df["label"] = label_enc.fit_transform(df["class_name"])

    feature_cols = [c for c in df.columns if c not in ("image_path", "class_name", "label")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    class_names = list(label_enc.classes_)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_means, inds[1])

    return X, y, feature_cols, class_names, df


def train_xgboost(X_train, y_train, X_test, y_test, num_classes):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, None

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    return model, y_pred


def train_lightgbm(X_train, y_train, X_test, y_test, num_classes):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None, None

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=num_classes,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = model.predict(X_test)
    return model, y_pred


def compute_shap(model, X, feature_names, output_dir: Path, model_name: str):
    try:
        import shap
    except ImportError:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:500])

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=1)

    importance = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    with open(output_dir / f"shap_importance_{model_name}.json", "w") as f:
        json.dump([{"feature": n, "importance": float(v)} for n, v in importance], f, indent=2)

    plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
    names = [x[0] for x in importance]
    vals = [x[1] for x in importance]
    plt.barh(range(len(names)), vals[::-1])
    plt.yticks(range(len(names)), names[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Feature Importance ({model_name})")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_importance_{model_name}.png", dpi=200)
    plt.close()
    print(f"  SHAP importance saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", type=str, required=True)
    parser.add_argument("--output", type=str, default="tabular_results")
    parser.add_argument("--splits-file", type=str, default=None,
                        help="Use same train/test split as baseline (data_splits.json)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols, class_names, df = load_data(args.attributes)
    num_classes = len(class_names)
    print(f"Loaded {len(X)} samples, {len(feature_cols)} features, {num_classes} classes")

    if args.splits_file:
        with open(args.splits_file) as f:
            splits = json.load(f)

        def basename_key(p):
            parts = Path(p).parts
            return parts[-2] + "/" + parts[-1] if len(parts) >= 2 else parts[-1]

        train_keys = {basename_key(s["path"]) for s in splits["train"]}
        val_keys = {basename_key(s["path"]) for s in splits["val"]}
        test_keys = {basename_key(s["path"]) for s in splits["test"]}

        df["_key"] = df["image_path"].apply(basename_key)
        train_mask = df["_key"].isin(train_keys | val_keys)
        test_mask = df["_key"].isin(test_keys)
        df.drop(columns=["_key"], inplace=True)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        print(f"Using splits from {args.splits_file}: train={len(X_train)}, test={len(X_test)}")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y)
        print(f"Random split: train={len(X_train)}, test={len(X_test)}")

    results = {}

    for name, train_fn in [("xgboost", train_xgboost), ("lightgbm", train_lightgbm)]:
        print(f"\nTraining {name}...")
        model, y_pred = train_fn(X_train, y_train, X_test, y_test, num_classes)
        if model is None:
            continue

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            "classification_report": classification_report(y_test, y_pred,
                                                           target_names=class_names, output_dict=True),
        }
        results[name] = metrics

        print(f"  accuracy:          {metrics['accuracy']:.4f}")
        print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  macro_f1:          {metrics['macro_f1']:.4f}")

        with open(output_dir / f"metrics_{name}.json", "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        compute_shap(model, X_test, feature_cols, output_dir, name)

    with open(output_dir / "comparison.json", "w") as f:
        summary = {name: {k: v for k, v in m.items() if k != "classification_report"}
                   for name, m in results.items()}
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
