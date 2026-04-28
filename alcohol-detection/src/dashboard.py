import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from prepare_data import window_features


def apply_theme(theme: str) -> None:
    dark = theme == "Dark"
    bg = "#0b1220" if dark else "#f5f7fb"
    panel = "#111827" if dark else "#ffffff"
    panel_soft = "#0f172a" if dark else "#eef2f7"
    text = "#e5e7eb" if dark else "#1f2937"
    mutetext = "#94a3b8" if dark else "#6b7280"
    border = "#1f2937" if dark else "#e5e7eb"
    accent = "#3b82f6"
    chart_grid = "#334155" if dark else "#cbd5e1"
    chart_text = "#cbd5e1" if dark else "#334155"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg};
            color: {text};
        }}
        p, label, span, div {{
            color: {text};
        }}
        .block-container {{
            max-width: 1320px;
            padding-top: 0.9rem;
        }}
        .hero {{
            border: 1px solid {border};
            border-radius: 8px;
            background: {panel};
            padding: 18px 20px;
            margin-bottom: 12px;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 1.9rem;
            font-weight: 750;
            color: {text};
        }}
        .hero p {{
            margin: 6px 0 0 0;
            font-size: 0.92rem;
            color: {mutetext};
        }}
        .kpi {{
            border: 1px solid {border};
            border-radius: 8px;
            background: {panel};
            padding: 10px 12px;
        }}
        .kpi .label {{
            color: {mutetext};
            font-size: 0.8rem;
        }}
        .kpi .value {{
            color: {text};
            font-size: 1.45rem;
            font-weight: 650;
            margin-top: 2px;
        }}
        .section {{
            border: 1px solid {border};
            border-radius: 8px;
            background: {panel};
            padding: 14px;
        }}
        .controlbar {{
            border: 1px solid {border};
            border-radius: 8px;
            background: {panel};
            padding: 12px 14px;
            margin-bottom: 12px;
        }}
        .section h3, .section h4, .section strong {{
            letter-spacing: 0;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {border};
        }}
        .stDownloadButton button, .stButton button {{
            border-radius: 8px !important;
            border: 1px solid {border} !important;
            font-weight: 600 !important;
        }}
        .subtle {{
            color: {mutetext};
            font-size: 0.86rem;
        }}
        [data-testid="stCaptionContainer"] * {{
            color: {mutetext} !important;
        }}
        [data-testid="stWidgetLabel"] * {{
            color: {text} !important;
        }}
        [data-baseweb="select"] * {{
            color: {text} !important;
        }}
        [data-baseweb="input"] * {{
            color: {text} !important;
        }}
        [data-baseweb="slider"] * {{
            color: {text} !important;
        }}
        [data-testid="stMarkdownContainer"] * {{
            color: inherit;
        }}
        .st-emotion-cache-1kyxreq {{
            color: {text};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px;
            background: {panel_soft};
            color: {text};
            border: 1px solid {border};
            padding: 8px 12px;
        }}
        .stTabs [aria-selected="true"] {{
            background: {accent}22;
            border-color: {accent};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    plt.rcParams.update(
        {
            "axes.facecolor": panel,
            "figure.facecolor": panel,
            "axes.edgecolor": chart_grid,
            "axes.labelcolor": chart_text,
            "xtick.color": chart_text,
            "ytick.color": chart_text,
            "text.color": chart_text,
            "axes.titleweight": "semibold",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "grid.color": chart_grid,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
        }
    )


def load_model_artifact(model_path: Path):
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], float(artifact.get("threshold", 0.5)), artifact.get("model_name", "model")
    return artifact, 0.5, "model"


def make_windows(df: pd.DataFrame, sample_hz: int, window_sec: int, overlap: float) -> pd.DataFrame:
    window_size = int(sample_hz * window_sec)
    step = max(1, int(window_size * (1.0 - overlap)))
    rows = []
    if len(df) < window_size:
        return pd.DataFrame()
    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        w = df.iloc[start:end]
        feats = window_features(w)
        feats["start_time"] = int(w["time"].iloc[0])
        feats["end_time"] = int(w["time"].iloc[-1])
        rows.append(feats)
    return pd.DataFrame(rows)


def predict_windows(model, threshold: float, Xw: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in Xw.columns if c not in {"start_time", "end_time"}]
    proba = model.predict_proba(Xw[feat_cols])[:, 1] if hasattr(model, "predict_proba") else model.predict(Xw[feat_cols]).astype(float)
    pred = (proba >= threshold).astype(int)
    out = Xw[["start_time", "end_time"]].copy()
    out["risk_score"] = proba
    out["pred_label"] = pred
    return out


def render_kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Alcohol Risk Dashboard", layout="wide")

base_dir = Path(__file__).resolve().parents[1]
default_data = base_dir / "data/raw/all_accelerometer_data_pids_13.csv"
default_model = base_dir / "models/best_model.joblib"
metrics_file = base_dir / "models/metrics.json"
comparison_file = base_dir / "reports/model_comparison.json"

if "theme_dark" not in st.session_state:
    st.session_state["theme_dark"] = False
apply_theme("Dark" if st.session_state["theme_dark"] else "Light")

if not default_model.exists():
    st.error(f"Missing model artifact: {default_model}")
    st.stop()

model, trained_threshold, model_name = load_model_artifact(default_model)

raw = pd.read_csv(default_data, usecols=["time", "pid", "x", "y", "z"])
raw["pid"] = raw["pid"].astype(str)
raw["time"] = pd.to_numeric(raw["time"], errors="coerce")
raw["x"] = pd.to_numeric(raw["x"], errors="coerce")
raw["y"] = pd.to_numeric(raw["y"], errors="coerce")
raw["z"] = pd.to_numeric(raw["z"], errors="coerce")
raw = raw.dropna(subset=["time", "pid", "x", "y", "z"]).copy()
raw["time"] = raw["time"].astype(np.int64)
raw = raw.sort_values(["pid", "time"])

st.markdown(
    f"""
    <div class="hero">
      <h1>Alchohol Risk Detection Dashboard</h1>
      <p>Single-view monitoring for participant-level inference, risk trends, and comparison analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="controlbar">', unsafe_allow_html=True)
ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.0, 0.8])
with ctrl1:
    pid_placeholder = st.empty()
with ctrl2:
    st.caption("Advanced settings are available below.")
with ctrl3:
    st.toggle("Dark", key="theme_dark")
st.markdown("</div>", unsafe_allow_html=True)

apply_theme("Dark" if st.session_state["theme_dark"] else "Light")

sample_hz = 40
window_sec = 10
overlap = 0.5
threshold = trained_threshold

with st.expander("Advanced Settings", expanded=False):
    adv1, adv2, adv3, adv4 = st.columns(4)
    with adv1:
        sample_hz = st.number_input("Sample Hz", min_value=1, max_value=200, value=40, step=1)
    with adv2:
        window_sec = st.number_input("Window Sec", min_value=1, max_value=60, value=10, step=1)
    with adv3:
        overlap = st.slider("Overlap", min_value=0.0, max_value=0.9, value=0.5, step=0.05)
    with adv4:
        use_override = st.checkbox("Override Threshold", value=False)
        threshold_override = st.slider("Threshold", min_value=0.05, max_value=0.95, value=float(trained_threshold), step=0.01)
    if use_override:
        threshold = threshold_override

pid_list = sorted(raw["pid"].unique().tolist())
with pid_placeholder:
    pid = st.selectbox("Participant ID", pid_list, index=0)
pid_df = raw[raw["pid"] == pid].copy()
windows = make_windows(pid_df, sample_hz=sample_hz, window_sec=window_sec, overlap=overlap)
if windows.empty:
    st.warning("Not enough rows for current window settings.")
    st.stop()
preds = predict_windows(model, threshold=threshold, Xw=windows)

summary_rows = []
for comp_pid, g in raw.groupby("pid"):
    w = make_windows(g, sample_hz=sample_hz, window_sec=window_sec, overlap=overlap)
    if w.empty:
        continue
    p = predict_windows(model, threshold=threshold, Xw=w)
    summary_rows.append(
        {
            "pid": comp_pid,
            "rows": len(g),
            "windows": len(p),
            "risk_rate": float(p["pred_label"].mean()),
            "mean_risk": float(p["risk_score"].mean()),
            "p95_risk": float(np.percentile(p["risk_score"], 95)),
        }
    )
summary = pd.DataFrame(summary_rows).sort_values("mean_risk", ascending=False).reset_index(drop=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_kpi("Model", model_name)
with c2:
    render_kpi("Threshold", f"{threshold:.3f}")
with c3:
    render_kpi("Selected PID Rows", f"{len(pid_df):,}")
with c4:
    render_kpi("PID High-Risk Rate", f"{preds['pred_label'].mean():.2%}")
with c5:
    render_kpi("Participants", f"{summary['pid'].nunique():,}")

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("**Model Comparison (Baseline vs 1D CNN)**")
if comparison_file.exists():
    with open(comparison_file, "r", encoding="utf-8") as f:
        cmp_data = json.load(f)
    row_baseline = cmp_data.get("baseline_logreg_features", {})
    row_cnn = cmp_data.get("cnn_1d_sequence", {})
    cmp_df = pd.DataFrame(
        [
            {
                "model": "Baseline LogReg Features",
                "f1": row_baseline.get("f1"),
                "precision": row_baseline.get("precision"),
                "recall": row_baseline.get("recall"),
                "roc_auc": row_baseline.get("roc_auc"),
            },
            {
                "model": "1D CNN Sequence",
                "f1": row_cnn.get("f1"),
                "precision": row_cnn.get("precision"),
                "recall": row_cnn.get("recall"),
                "roc_auc": row_cnn.get("roc_auc"),
            },
        ]
    )
    mleft, mright = st.columns([1, 1.3])
    with mleft:
        st.dataframe(cmp_df, use_container_width=True, height=140)
    with mright:
        figm, axm = plt.subplots(figsize=(6.5, 2.6))
        metrics = ["f1", "precision", "recall", "roc_auc"]
        x = np.arange(len(metrics))
        width = 0.36
        axm.bar(x - width / 2, [row_baseline.get(m, 0) for m in metrics], width, label="Baseline", color="#64748b")
        axm.bar(x + width / 2, [row_cnn.get(m, 0) for m in metrics], width, label="CNN", color="#2563eb")
        axm.set_xticks(x)
        axm.set_xticklabels(metrics)
        axm.set_ylim(0, 1.0)
        axm.grid(axis="y", alpha=0.2)
        axm.legend(frameon=False, loc="upper left")
        st.pyplot(figm, use_container_width=True)
else:
    st.info("Run `python src/train_sequence.py --max-windows-per-pid 500` to generate model comparison.")
st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("**Most Recent Performance**")
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    primary = "#60a5fa" if st.session_state["theme_dark"] else "#2563eb"
    danger = "#f87171" if st.session_state["theme_dark"] else "#dc2626"
    ax.plot(preds["start_time"], preds["risk_score"], color=primary, linewidth=1.9)
    ax.fill_between(preds["start_time"], preds["risk_score"], alpha=0.16, color=primary)
    ax.axhline(y=threshold, linestyle="--", color=danger, linewidth=1.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Risk Score")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("**Best Performers (Lowest Mean Risk)**")
    best = summary.sort_values("mean_risk", ascending=True).head(8)
    fig2, ax2 = plt.subplots(figsize=(6.6, 3.4))
    success = "#34d399" if st.session_state["theme_dark"] else "#059669"
    ax2.plot(best["pid"], best["mean_risk"], marker="o", color=success, linewidth=1.9)
    ax2.set_xlabel("PID")
    ax2.set_ylabel("Mean Risk")
    ax2.tick_params(axis="x", rotation=35)
    ax2.grid(alpha=0.2)
    st.pyplot(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

left2, right2 = st.columns(2)
with left2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("**Participant Comparison Table**")
    st.dataframe(summary, use_container_width=True, height=330)
    st.markdown("</div>", unsafe_allow_html=True)
with right2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("**Worst Performers (Highest Mean Risk)**")
    worst = summary.sort_values("mean_risk", ascending=False).head(8)
    fig3, ax3 = plt.subplots(figsize=(6.6, 3.4))
    danger = "#f87171" if st.session_state["theme_dark"] else "#dc2626"
    ax3.plot(worst["pid"], worst["mean_risk"], marker="o", color=danger, linewidth=1.9)
    ax3.set_xlabel("PID")
    ax3.set_ylabel("Mean Risk")
    ax3.tick_params(axis="x", rotation=35)
    ax3.grid(alpha=0.2)
    st.pyplot(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("**Selected Participant Predictions**")
st.dataframe(preds.head(150), use_container_width=True, height=280)
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "Download Selected PID Predictions",
        data=preds.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{pid}.csv",
        mime="text/csv",
    )
with dl2:
    st.download_button(
        "Download Participant Summary",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="participant_comparison.csv",
        mime="text/csv",
    )
st.markdown("</div>", unsafe_allow_html=True)
