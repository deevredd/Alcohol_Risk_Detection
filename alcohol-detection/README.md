# Alchohol Risk Detection Dashboard

Sensor-driven heavy-drinking risk detection using smartphone accelerometer time-series and TAC-based labels (UCI Bar Crawl dataset), with an interactive analytics dashboard.

## Project Scope

- Sliding-window signal processing pipeline for accelerometer streams.
- Classical ML benchmark (RandomForest / ExtraTrees / LogisticRegression / HistGB / XGBoost).
- Sequence-model benchmark (1D CNN, PyTorch) with subject-aware split.
- Single-page web dashboard for risk trends, participant comparison, and downloads.

## Dataset

- UCI Bar Crawl: Detecting Heavy Drinking
- Expected archive path: `bar+crawl+detecting+heavy+drinking.zip`

## Environment

- Python: 3.9 (recommended for this setup)
- Install dependencies:

```bash
pip install -r requirements.txt
```

For sequence model training (1D CNN), install the extra dependency:

```bash
pip install -r requirements-sequence.txt
```

## Run End-to-End

```bash
python src/prepare_data.py
python src/train.py
python src/evaluate.py
```

## Sequence Benchmark

```bash
python src/train_sequence.py --max-windows-per-pid 500
```

## Launch Dashboard

```bash
python -m streamlit run src/dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Open: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## Deployment

### Streamlit Community Cloud

1. Push this project to GitHub.
2. Create a new Streamlit app.
3. Set entry file to `app.py`.
4. Deploy with default `requirements.txt`.

### Render (Web Service)

Use included `Procfile`:

```text
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## Current Results Snapshot

### Classical holdout (`reports/test_metrics.json`)

| Metric | Value |
|---|---:|
| F1 | 0.6686 |
| Precision | 0.5975 |
| Recall | 0.7588 |
| ROC-AUC | 0.6950 |

### Sequence vs Baseline (`reports/model_comparison.json`, `--max-windows-per-pid 500`)

| Model | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline LogReg Features | 0.4925 | 0.3530 | 0.8142 | 0.5448 |
| 1D CNN Sequence | 0.5179 | 0.3706 | 0.8596 | 0.5298 |

## Repo Structure

```text
alcohol-detection/
  data/
    raw/
    processed/
  models/
  reports/
  src/
    prepare_data.py
    train.py
    evaluate.py
    train_sequence.py
    dashboard.py
  requirements.txt
  README.md
```
