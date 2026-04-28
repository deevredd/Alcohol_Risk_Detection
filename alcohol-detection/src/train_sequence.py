import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_data import load_accelerometer, load_tac


def build_sequence_dataset(
    acc_df: pd.DataFrame,
    tac_df: pd.DataFrame,
    sample_hz: int,
    window_sec: int,
    overlap: float,
    tac_threshold: float,
    sustain_ratio: float,
    max_windows_per_pid: int,
):
    window_size = int(sample_hz * window_sec)
    step = max(1, int(window_size * (1.0 - overlap)))

    seqs, labels, groups = [], [], []
    rows = []
    for pid, g in acc_df.groupby("pid"):
        g = g.sort_values("time").reset_index(drop=True)
        for c in ["x", "y", "z"]:
            mu = float(g[c].mean())
            sd = float(g[c].std())
            g[c] = (g[c] - mu) / sd if sd > 0 else (g[c] - mu)

        tac_g = tac_df[tac_df["pid"] == pid][["time", "tac"]].sort_values("time")
        if tac_g.empty:
            continue
        g = g[(g["time"] >= tac_g["time"].min()) & (g["time"] <= tac_g["time"].max())].copy()
        if len(g) < window_size:
            continue

        tac_interp = np.interp(
            g["time"].to_numpy(dtype=np.float64),
            tac_g["time"].to_numpy(dtype=np.float64),
            tac_g["tac"].to_numpy(dtype=np.float64),
        )
        g["tac"] = tac_interp
        mag = np.sqrt(g["x"] ** 2 + g["y"] ** 2 + g["z"] ** 2)
        g["mag"] = mag

        count = 0
        for start in range(0, len(g) - window_size + 1, step):
            w = g.iloc[start : start + window_size]
            high_ratio = float((w["tac"] >= tac_threshold).mean())
            label = 1 if high_ratio >= sustain_ratio else 0
            seq = w[["x", "y", "z", "mag"]].to_numpy(dtype=np.float32)
            seqs.append(seq)
            labels.append(label)
            groups.append(pid)
            rows.append(
                {
                    "x_mean": float(w["x"].mean()),
                    "x_std": float(w["x"].std()),
                    "y_mean": float(w["y"].mean()),
                    "y_std": float(w["y"].std()),
                    "z_mean": float(w["z"].mean()),
                    "z_std": float(w["z"].std()),
                    "mag_mean": float(w["mag"].mean()),
                    "mag_std": float(w["mag"].std()),
                    "mag_p95": float(np.percentile(w["mag"], 95)),
                    "label": label,
                    "pid": pid,
                }
            )
            count += 1
            if max_windows_per_pid > 0 and count >= max_windows_per_pid:
                break

    X_seq = np.stack(seqs) if seqs else np.empty((0, window_size, 4), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    X_tab = pd.DataFrame(rows)
    return X_seq, y, groups, X_tab


class CNN1D(nn.Module):
    def __init__(self, n_channels: int, seq_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_cnn(X_train, y_train, X_val, y_val, epochs=25, batch_size=256, lr=3e-4, patience=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(n_channels=X_train.shape[2], seq_len=X_train.shape[1]).to(device)
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_train).permute(0, 2, 1),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_x = torch.tensor(X_val).permute(0, 2, 1).to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state, best_auc, best_val_loss = None, -1.0, float("inf")
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(val_x)
            probs = torch.sigmoid(logits_val).detach().cpu().numpy()
            val_loss = float(
                criterion(logits_val, torch.tensor(y_val, dtype=torch.float32, device=device)).item()
            )
        try:
            auc = roc_auc_score(y_val, probs)
        except ValueError:
            auc = np.nan

        improved = False
        if np.isfinite(auc) and auc > best_auc:
            improved = True
        elif not np.isfinite(auc) and val_loss < best_val_loss:
            improved = True

        if improved:
            best_auc = auc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        auc_text = f"{auc:.4f}" if np.isfinite(auc) else "nan"
        print(f"Epoch {epoch + 1}/{epochs} val_auc={auc_text} val_loss={val_loss:.4f}")
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1} (best val_auc={best_auc:.4f})")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_probs(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-path", type=Path, default=Path("reports/model_comparison.json"))
    parser.add_argument("--sample-hz", type=int, default=40)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--tac-threshold", type=float, default=0.08)
    parser.add_argument("--sustain-ratio", type=float, default=0.6)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-windows-per-pid", type=int, default=5000)
    args = parser.parse_args()

    random.seed(args.random_state)
    np.random.seed(args.random_state)
    os.environ["PYTHONHASHSEED"] = str(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)

    print("Loading data...")
    acc = load_accelerometer(args.raw_dir / "all_accelerometer_data_pids_13.csv")
    tac = load_tac(args.raw_dir / "clean_tac")
    print("Building sequence dataset...")
    X_seq, y, groups, X_tab = build_sequence_dataset(
        acc,
        tac,
        sample_hz=args.sample_hz,
        window_sec=args.window_sec,
        overlap=args.overlap,
        tac_threshold=args.tac_threshold,
        sustain_ratio=args.sustain_ratio,
        max_windows_per_pid=args.max_windows_per_pid,
    )
    print(f"Dataset ready: seq={X_seq.shape}, tab={X_tab.shape}")
    feature_cols = [c for c in X_tab.columns if c not in {"label", "pid"}]
    y_tab = X_tab["label"].to_numpy(dtype=np.int64)

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    tr_idx, te_idx = next(gss.split(X_seq, y, groups=groups))

    # Create a validation split from train only, grouped by participant.
    train_groups = groups[tr_idx]
    tr_inner_idx, va_inner_idx = None, None
    for seed in range(args.random_state, args.random_state + 20):
        inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_rel, va_rel = next(inner_gss.split(X_seq[tr_idx], y[tr_idx], groups=train_groups))
        y_tr_try = y[tr_idx][tr_rel]
        y_va_try = y[tr_idx][va_rel]
        if len(np.unique(y_tr_try)) > 1 and len(np.unique(y_va_try)) > 1:
            tr_inner_idx = tr_idx[tr_rel]
            va_inner_idx = tr_idx[va_rel]
            break
    if tr_inner_idx is None:
        # Fallback without grouping to guarantee both classes in val.
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
        tr_rel, va_rel = next(sss.split(X_seq[tr_idx], y[tr_idx]))
        tr_inner_idx = tr_idx[tr_rel]
        va_inner_idx = tr_idx[va_rel]

    # Classical baseline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1500)),
        ]
    )
    pipe.fit(X_tab.iloc[tr_idx][feature_cols], y_tab[tr_idx])
    tab_probs = pipe.predict_proba(X_tab.iloc[te_idx][feature_cols])[:, 1]
    tab_metrics = evaluate_probs(y_tab[te_idx], tab_probs, threshold=0.5)

    # Sequence model
    print("Training 1D CNN...")
    cnn = fit_cnn(X_seq[tr_inner_idx], y[tr_inner_idx], X_seq[va_inner_idx], y[va_inner_idx])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        probs = torch.sigmoid(
            cnn(torch.tensor(X_seq[te_idx]).permute(0, 2, 1).to(device))
        ).cpu().numpy()
    cnn_metrics = evaluate_probs(y[te_idx], probs, threshold=0.5)

    results = {
        "split": {"train_size": int(len(tr_idx)), "test_size": int(len(te_idx)), "participants": int(len(np.unique(groups)))},
        "baseline_logreg_features": tab_metrics,
        "cnn_1d_sequence": cnn_metrics,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
    print("Training classical baseline...")
