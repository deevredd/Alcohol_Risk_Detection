import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(args.processed_dir / "X.csv")
    y = pd.read_csv(args.processed_dir / "y.csv")["label"]
    groups = pd.read_csv(args.processed_dir / "groups.csv")["pid"].astype(str)

    artifact = joblib.load(args.models_dir / "best_model.joblib")
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        threshold = float(artifact.get("threshold", 0.5))
    else:
        model = artifact
        threshold = 0.5

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Refit a fresh estimator on train split only to avoid leakage.
    eval_model = clone(model)
    eval_model.fit(X_train, y_train)

    if hasattr(eval_model, "predict_proba"):
        proba = eval_model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
    else:
        pred = eval_model.predict(X_test)
        proba = pred

    metrics = {
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    with open(args.reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.reports_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.reports_dir / "roc_curve.png", dpi=180)
    plt.close()

    print("Test metrics:")
    print(metrics)
    print(f"Saved: {args.reports_dir / 'test_metrics.json'}")
    print(f"Saved: {args.reports_dir / 'confusion_matrix.png'}")
    print(f"Saved: {args.reports_dir / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
