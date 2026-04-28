import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def load_accelerometer(acc_path: Path) -> pd.DataFrame:
    df = pd.read_csv(acc_path)
    required = {"time", "pid", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required accelerometer columns: {missing}")
    df = df[list(required)].copy()
    df = df.dropna(subset=["time", "pid", "x", "y", "z"])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].astype(np.int64)
    for c in ["x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y", "z"])
    df["pid"] = df["pid"].astype(str)
    df = df.sort_values(["pid", "time"]).reset_index(drop=True)
    return df


def load_tac(clean_tac_dir: Path) -> pd.DataFrame:
    rows = []
    tac_files = sorted(clean_tac_dir.glob("*.csv"))
    if not tac_files:
        raise FileNotFoundError(f"No TAC CSV files found in: {clean_tac_dir}")

    for fp in tac_files:
        pid = fp.stem
        if pid.endswith("_clean_TAC"):
            pid = pid.replace("_clean_TAC", "")
        tdf = pd.read_csv(fp)
        cols = {c.lower(): c for c in tdf.columns}
        if "timestamp" in cols:
            t_col = cols["timestamp"]
        elif "time" in cols:
            t_col = cols["time"]
        else:
            raise ValueError(f"Could not find time column in {fp.name}")
        tac_candidates = [c for c in tdf.columns if "tac" in c.lower()]
        if "tac" in cols:
            tac_col = cols["tac"]
        elif tac_candidates:
            tac_col = tac_candidates[0]
        else:
            raise ValueError(f"Could not find TAC column in {fp.name}")
        use = tdf[[t_col, tac_col]].copy()
        use.columns = ["time", "tac"]
        use["time"] = pd.to_numeric(use["time"], errors="coerce")
        use["tac"] = pd.to_numeric(use["tac"], errors="coerce")
        use = use.dropna(subset=["time", "tac"])
        use["time"] = use["time"].astype(np.int64)
        # TAC timestamps in this dataset are in Unix seconds; convert to ms.
        if use["time"].max() < 10_000_000_000:
            use["time"] = use["time"] * 1000
        use["pid"] = str(pid)
        rows.append(use)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["pid", "time"]).reset_index(drop=True)
    return out


def window_features(wdf: pd.DataFrame) -> dict:
    feats = {}
    axes = ["x", "y", "z"]
    mag = np.sqrt((wdf["x"] ** 2) + (wdf["y"] ** 2) + (wdf["z"] ** 2))
    for c in axes:
        vals = wdf[c].to_numpy()
        feats[f"{c}_mean"] = float(np.mean(vals))
        feats[f"{c}_std"] = float(np.std(vals))
        feats[f"{c}_min"] = float(np.min(vals))
        feats[f"{c}_max"] = float(np.max(vals))
        feats[f"{c}_energy"] = float(np.mean(vals**2))
        feats[f"{c}_p25"] = float(np.percentile(vals, 25))
        feats[f"{c}_p75"] = float(np.percentile(vals, 75))
        feats[f"{c}_iqr"] = feats[f"{c}_p75"] - feats[f"{c}_p25"]
        feats[f"{c}_mad"] = float(np.mean(np.abs(vals - np.mean(vals))))
        fft_vals = np.abs(np.fft.rfft(vals - np.mean(vals)))
        feats[f"{c}_fft_mean"] = float(np.mean(fft_vals))
        feats[f"{c}_fft_std"] = float(np.std(fft_vals))
    feats["mag_mean"] = float(np.mean(mag))
    feats["mag_std"] = float(np.std(mag))
    feats["mag_min"] = float(np.min(mag))
    feats["mag_max"] = float(np.max(mag))
    feats["mag_energy"] = float(np.mean(mag**2))
    feats["mag_p25"] = float(np.percentile(mag, 25))
    feats["mag_p75"] = float(np.percentile(mag, 75))
    feats["mag_iqr"] = feats["mag_p75"] - feats["mag_p25"]
    feats["mag_mad"] = float(np.mean(np.abs(mag - np.mean(mag))))

    def safe_corr(a: pd.Series, b: pd.Series) -> float:
        if len(a) < 2:
            return 0.0
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    feats["xy_corr"] = safe_corr(wdf["x"], wdf["y"])
    feats["xz_corr"] = safe_corr(wdf["x"], wdf["z"])
    feats["yz_corr"] = safe_corr(wdf["y"], wdf["z"])
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
    return feats


def build_dataset(
    acc_df: pd.DataFrame,
    tac_df: pd.DataFrame,
    sample_hz: int,
    window_sec: int,
    overlap: float,
    tac_threshold: float,
    sustain_ratio: float,
    normalize_per_pid: bool,
) -> pd.DataFrame:
    window_size = int(sample_hz * window_sec)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    step = max(1, int(window_size * (1.0 - overlap)))

    all_rows = []
    for pid, g in acc_df.groupby("pid"):
        g = g.sort_values("time").reset_index(drop=True)
        if normalize_per_pid:
            for c in ["x", "y", "z"]:
                mu = float(g[c].mean())
                sd = float(g[c].std())
                if sd > 0:
                    g[c] = (g[c] - mu) / sd
                else:
                    g[c] = g[c] - mu
        tac_g = tac_df[tac_df["pid"] == pid][["time", "tac"]].sort_values("time")
        if tac_g.empty:
            continue

        # TAC readings are sparse; interpolate onto accelerometer timeline.
        g = g[(g["time"] >= tac_g["time"].min()) & (g["time"] <= tac_g["time"].max())].copy()
        if g.empty:
            continue

        tac_interp = np.interp(
            g["time"].to_numpy(dtype=np.float64),
            tac_g["time"].to_numpy(dtype=np.float64),
            tac_g["tac"].to_numpy(dtype=np.float64),
        )
        merged = g.copy()
        merged["tac"] = tac_interp
        if len(merged) < window_size:
            continue

        for start in range(0, len(merged) - window_size + 1, step):
            end = start + window_size
            w = merged.iloc[start:end]
            feats = window_features(w)
            tac_val = float(w["tac"].mean())
            tac_high_ratio = float((w["tac"] >= tac_threshold).mean())
            tac_slope = float(np.polyfit(np.arange(len(w), dtype=np.float64), w["tac"].to_numpy(), 1)[0]) if len(w) > 1 else 0.0
            label = 1 if tac_high_ratio >= sustain_ratio else 0
            feats["pid"] = pid
            feats["start_time"] = int(w["time"].iloc[0])
            feats["end_time"] = int(w["time"].iloc[-1])
            feats["tac_mean"] = tac_val
            feats["tac_high_ratio"] = tac_high_ratio
            feats["tac_slope"] = tac_slope
            feats["label"] = label
            all_rows.append(feats)

    if not all_rows:
        raise RuntimeError("No windows generated. Check input files and parameters.")
    return pd.DataFrame(all_rows)


def ensure_dataset_extracted(zip_path: Path, raw_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    acc_csv = raw_dir / "all_accelerometer_data_pids_13.csv"
    tac_dir = raw_dir / "clean_tac"
    if acc_csv.exists() and tac_dir.exists() and any(tac_dir.glob("*.csv")):
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())
        if "data.zip" in names:
            z.extract("data.zip", path=raw_dir)
            nested = raw_dir / "data.zip"
            with zipfile.ZipFile(nested, "r") as nz:
                nz.extractall(path=raw_dir)
        else:
            z.extractall(path=raw_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("bar+crawl+detecting+heavy+drinking.zip"),
        help="Outer zip path. If CSVs are missing in raw-dir, this zip will be extracted.",
    )
    parser.add_argument("--acc-file", type=str, default="all_accelerometer_data_pids_13.csv")
    parser.add_argument("--tac-dir", type=str, default="clean_tac")
    parser.add_argument("--sample-hz", type=int, default=40)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--tac-threshold", type=float, default=0.08)
    parser.add_argument("--sustain-ratio", type=float, default=0.6)
    parser.add_argument("--normalize-per-pid", action="store_true", default=True)
    parser.add_argument("--no-normalize-per-pid", dest="normalize_per_pid", action="store_false")
    args = parser.parse_args()

    ensure_dataset_extracted(args.zip_path, args.raw_dir)
    acc_path = args.raw_dir / args.acc_file
    tac_path = args.raw_dir / args.tac_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading accelerometer data: {acc_path}")
    acc_df = load_accelerometer(acc_path)
    print(f"Loading TAC data: {tac_path}")
    tac_df = load_tac(tac_path)

    print("Building windowed feature dataset...")
    df = build_dataset(
        acc_df=acc_df,
        tac_df=tac_df,
        sample_hz=args.sample_hz,
        window_sec=args.window_sec,
        overlap=args.overlap,
        tac_threshold=args.tac_threshold,
        sustain_ratio=args.sustain_ratio,
        normalize_per_pid=args.normalize_per_pid,
    )

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "label",
            "pid",
            "start_time",
            "end_time",
            "tac_mean",
            "tac_high_ratio",
            "tac_slope",
        }
    ]
    X = df[feature_cols].copy()
    y = df[["label"]].copy()
    groups = df[["pid"]].copy()
    meta = df[["pid", "start_time", "end_time", "tac_mean", "label"]].copy()

    X.to_csv(args.out_dir / "X.csv", index=False)
    y.to_csv(args.out_dir / "y.csv", index=False)
    groups.to_csv(args.out_dir / "groups.csv", index=False)
    meta.to_csv(args.out_dir / "meta.csv", index=False)

    print(f"Saved X: {args.out_dir / 'X.csv'} shape={X.shape}")
    print(f"Saved y: {args.out_dir / 'y.csv'} shape={y.shape}")
    print(f"Saved groups: {args.out_dir / 'groups.csv'} shape={groups.shape}")
    print(f"Saved meta: {args.out_dir / 'meta.csv'} shape={meta.shape}")


if __name__ == "__main__":
    main()
