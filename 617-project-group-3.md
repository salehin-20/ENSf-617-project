# 617 Project  Group 3 Runbook

This file captures the steps, commands, and notes to reproduce our NYISO load + weather forecasting project (LSTM baseline and TFT). All paths are relative to the repo root.

## Environment
- Python: 3.12 (virtualenv in `.venv`)
- Main deps: pytorch, pytorch-lightning, pytorch-forecasting, pandas, numpy, matplotlib, scikit-learn, holidays, requests, yaml.

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
- Config: `src/config.yaml` (timezone, paths, splits, quantiles, extremes).
- Raw/processed dirs: `data/raw/`, `data/processed/`.
- Data pull script: `scripts/data_pull.py`
  - Uses EIA v2 (series EBA.NYIS-ALL.D.H) if `EIA_API_KEY` is set; otherwise falls back to NYISO MIS CSVs.
  - Weather from Open-Meteo archive (NYC), yearly chunks.
  - Adds holiday flag + extreme_flag (temp outside configured percentiles).
  - Output: `data/processed/all.parquet` (tz America/New_York).
  - Flags:
    - `--force` rebuilds even if output exists.
    - `--refresh-weather` forces re-download of weather (otherwise cached `raw/weather/weather_raw.parquet`).

Example fresh pull:
```bash
source .venv/bin/activate
export EIA_API_KEY=3964PjpjKcuYZ1Qs8qjZWmblIg21yfCdZ0rqewKO
python scripts/data_pull.py --force --refresh-weather
```

## Models
### LSTM Baseline
- Code: `src/models/lstm_baseline.py`
- Train: `scripts/train_lstm.py`
- Eval: `scripts/eval_lstm.py`
- Features: y, temp, holiday, extreme_flag. Normalizes y/temp on train split.
- Available checkpoints:

| Version | Checkpoint | Epochs |
|---|---|---|
| version_2 | `epoch=4-step=1705.ckpt` | 5 |
| version_3 | `epoch=0-step=342.ckpt` | 1 |
| version_4 | `epoch=4-step=3410.ckpt` | 5 (best — use this) |

- Example eval using best checkpoint (version_4):
```bash
python scripts/eval_lstm.py --checkpoint reports/lstm_baseline/lightning_logs/version_4/checkpoints/epoch=4-step=3410.ckpt
```
- Reference metrics (normalized units, horizon 24): MAE 0.279, RMSE 0.422, MAPE 2.26%, pinball 0.091.

### Temporal Fusion Transformer (TFT)
- Train: `scripts/train_tft.py`
  - Uses TimeSeriesDataSet with GroupNormalizer, lookback=336 (14 days), horizon=24, allow_missing_timesteps=True.
  - Checkpoints saved under `reports/tft/lightning_logs/version_X/`.
- Eval: `scripts/eval_tft.py`
  - Required: `--checkpoint` path.
  - Options: `--device cpu|cuda`, `--batch_size`, `--num_workers`, `--max_batches` (debug), `--save_pred`, `--save_metrics`.
  - Outputs:
    - Metrics YAML: `reports/tft/metrics.yaml`
    - Full preds (denormed) Parquet: `reports/tft/preds.parquet` with columns ds, y, p10, p50, p90.
- Available checkpoints:

| Version | Checkpoint | Epochs |
|---|---|---|
| version_3 | `epoch=14-step=10170.ckpt` | 15 |
| version_4 | `epoch=19-step=13560.ckpt` | 20 (best — use this) |
| version_5 | `epoch=4-step=3390.ckpt` | 5 |

- Latest TFT metrics (from metrics.yaml): MAE 835.803, RMSE 1289.662, MAPE 6.20%, pinball 585.909 (horizon 24, lookback 336).

Run eval on GPU using best checkpoint (version_4):
```bash
python scripts/eval_tft.py \
  --checkpoint reports/tft/lightning_logs/version_4/checkpoints/epoch=19-step=13560.ckpt \
  --device cuda --batch_size 256 --num_workers 0
```

## Plots & Reporting
- Plot script: `scripts/report_plots.py`
  - Inputs: preds parquet (default `reports/tft/preds.parquet`), config.
  - Outputs (in `reports/`):
    - `calibration_tft.png` (nominal vs empirical coverage)
    - `sample_day_tft.png` (latest day forecast with P10/P90 band)
    - `extreme_mae_tft.png` (MAE normal vs extreme)

Generate plots:
```bash
python scripts/report_plots.py
```

- Comparison table: `reports/comparison.md`
  - LSTM baseline metrics (normalized) and TFT metrics (denormed) are recorded here.

## Typical end-to-end run
```bash
source .venv/bin/activate
python scripts/data_pull.py --force --refresh-weather
python scripts/train_lstm.py       # optional retrain
python scripts/train_tft.py --max_epochs 15  # adjust as needed
python scripts/eval_tft.py --checkpoint reports/tft/lightning_logs/version_3/checkpoints/epoch=14-step=10170.ckpt --device cuda --batch_size 256 --num_workers 0
python scripts/report_plots.py
cat reports/comparison.md
```

## Notes & gotchas
- Ensure timezone awareness: ds is tz-aware (America/New_York); time_idx is hours since epoch.
- TFT outputs from pytorch-forecasting are already de-normalized; eval script uses them directly.
- MAPE denominator is clamped to avoid blow-up on very small targets.
- GPU available: NVIDIA GeForce RTX 5060 Laptop GPU in our WSL setup. Use `--device cuda` for speed.
- Cached weather speeds reruns; use `--refresh-weather` only when needed.

## File map (key ones)
- `scripts/data_pull.py` � data download/merge/features
- `scripts/train_lstm.py`, `scripts/eval_lstm.py`
- `scripts/train_tft.py`, `scripts/eval_tft.py`
- `scripts/report_plots.py`
- `src/models/lstm_baseline.py`
- `src/config.yaml`
- Outputs: `data/processed/all.parquet`, `reports/tft/preds.parquet`, `reports/tft/metrics.yaml`, `reports/*.png`, `reports/comparison.md`
