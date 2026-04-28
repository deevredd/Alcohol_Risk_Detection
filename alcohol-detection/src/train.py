import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 33)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def evaluate_cv_with_threshold(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    f1s, prs, rcs, aucs, ths = [], [], [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fit_model = clone(model)
        fit_model.fit(X_train, y_train)
        train_proba = fit_model.predict_proba(X_train)[:, 1]
        t = best_threshold(y_train.to_numpy(), train_proba)
        test_proba = fit_model.predict_proba(X_test)[:, 1]
        pred = (test_proba >= t).astype(int)

        f1s.append(f1_score(y_test, pred, zero_division=0))
        prs.append(precision_score(y_test, pred, zero_division=0))
        rcs.append(recall_score(y_test, pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_test, test_proba))
        except ValueError:
            aucs.append(np.nan)
        ths.append(t)

    return {
        "f1": float(np.nanmean(f1s)),
        "precision": float(np.nanmean(prs)),
        "recall": float(np.nanmean(rcs)),
        "roc_auc": float(np.nanmean(aucs)),
        "threshold": float(np.nanmean(ths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    args.models_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(args.processed_dir / "X.csv")
    y = pd.read_csv(args.processed_dir / "y.csv")["label"]
    groups = pd.read_csv(args.processed_dir / "groups.csv")["pid"].astype(str)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=320,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=450,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "logreg_scaled": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=args.seed)),
            ]
        ),
        "hist_gb": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=10,
            max_iter=450,
            min_samples_leaf=40,
            random_state=args.seed,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            reg_alpha=0.2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
        )

    results = {}
    best_name, best_score, best_model, best_threshold_value = None, -1.0, None, 0.5

    for name, model in models.items():
        print(f"Training/Evaluating: {name}")
        metrics = evaluate_cv_with_threshold(model, X, y, groups, n_splits=5)
        results[name] = metrics
        print(f"{name} -> {metrics}")
        score = metrics["f1"] + 0.25 * max(0.0, metrics["roc_auc"] - 0.5)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
            best_threshold_value = metrics["threshold"]

    print(f"Best model by composite score: {best_name} ({best_score:.4f})")
    best_model.fit(X, y)

    artifact = {
        "model": best_model,
        "threshold": best_threshold_value,
        "model_name": best_name,
    }
    joblib.dump(artifact, args.models_dir / "best_model.joblib")

    with open(args.models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_results": results,
                "best_model": best_name,
                "best_threshold": best_threshold_value,
            },
            f,
            indent=2,
        )

    print(f"Saved model artifact: {args.models_dir / 'best_model.joblib'}")
    print(f"Saved metrics: {args.models_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
