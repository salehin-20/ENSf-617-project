# 617-project — Energy Load Forecasting

Group project (4 members) to predict day-ahead electricity demand using an LSTM baseline vs. a Temporal Fusion Transformer (TFT), with uncertainty estimates and stress tests on extreme weather days.

## Quick start
1. **Create env & install**
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. **Data layout**
   - `data/raw/` — downloads from load + weather sources (not tracked)
   - `data/processed/` — joined/cleaned features (not tracked)
3. **Data pull**
   - Use `notebooks/01_data_pull.ipynb` (to be added) or `scripts/data_pull.py` to download:
     - Load: chosen region (e.g., NYISO system load)
     - Weather: Meteostat hourly for nearest stations
   - Save a single parquet/CSV with aligned timestamps, holiday flags, and extreme-weather flags.
4. **Training**
   - Baseline LSTM: `src/models/lstm_baseline.py` (todo)
   - TFT: `src/models/tft.py` (todo)
   - Track runs with MLflow or W&B; metrics: MAE, RMSE, MAPE, pinball loss, coverage.
5. **Evaluation**
   - Time-based train/val/test (hold out most recent year).
   - Extra slice for extreme weather (e.g., top/bottom 5% temperature).
   - Plots/tables land in `reports/`.

## Proposed folders
```
data/
  raw/
  processed/
notebooks/
  01_data_pull.ipynb
  02_eda.ipynb
src/
  data/
  features/
  models/
  utils/
reports/
scripts/
```

## Suggested roles
- Data scout
- Baseline builder (LSTM + simple stats)
- TFT specialist
- Evaluator/storyteller (metrics, slices, report)

## Next actions
- Pick region + date range; fix timezone.
- Commit the data pull notebook/script.
- Define train/val/test splits and extreme-weather thresholds in a shared config.


## How to Run (WSL/Linux)

1) Python + venv
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip python-is-python3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # installs torch & friends
```
If `torch` is missing later, reinstall via CPU wheel:
`pip install --index-url https://download.pytorch.org/whl/cpu torch`

2) Optional: EIA API key for load data
`export EIA_API_KEY=3964PjpjKcuYZ1Qs8qjZWmblIg21yfCdZ0rqewKO`

3) Pull data (rebuild + refresh caches)
```bash
python scripts/data_pull.py --force --refresh-weather
```

4) Train models
```bash
# TFT (produces checkpoints under reports/tft/lightning_logs/)
python scripts/train_tft.py --max_epochs 15 --batch_size 64
# LSTM baseline
python scripts/train_lstm.py --max_epochs 5 --batch_size 128 --lookback 24 --horizon 24
```

5) Evaluate TFT (pick an existing checkpoint)
Available ckpts on disk:
- reports/tft/lightning_logs/version_3/checkpoints/epoch=14-step=10170.ckpt
- reports/tft/lightning_logs/version_4/checkpoints/epoch=19-step=13560.ckpt
- reports/tft/lightning_logs/version_5/checkpoints/epoch=4-step=3390.ckpt
Example:
```bash
python scripts/eval_tft.py --checkpoint reports/tft/lightning_logs/version_4/checkpoints/epoch=19-step=13560.ckpt
```
This writes reports/tft/preds.csv and reports/tft/metrics.yaml.

6) Plots (uses latest preds):
```bash
python scripts/report_plots.py
```
Outputs: reports/calibration_tft.png, reports/sample_day_tft.png, reports/extreme_mae_tft.png.

Common pitfalls we hit:
- python not found ? install python3 and python-is-python3 (step 1).
- torch ModuleNotFoundError ? reinstall via PyTorch index (see step 1 note).
- FileNotFoundError on checkpoints ? use real ckpts above (not placeholder version_0 paths).

Notebook: open notebooks/notebook.ipynb for walkthrough.
Runbook: see 617-project-group-3.md for full details.

