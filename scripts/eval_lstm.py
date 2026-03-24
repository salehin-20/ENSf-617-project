import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.lstm_baseline import LSTMBaseline, LSTMConfig


class EvalWindowDataset(Dataset):
    def __init__(self, norm_arr: np.ndarray, raw_arr: np.ndarray, lookback: int, horizon: int, feature_cols: slice, extreme_col: int):
        self.norm_arr = norm_arr
        self.raw_arr = raw_arr
        self.lookback = lookback
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.extreme_col = extreme_col
        self.n = norm_arr.shape[0] - lookback - horizon + 1

    def __len__(self):
        return max(self.n, 0)

    def __getitem__(self, idx):
        start = idx
        mid = start + self.lookback
        end = mid + self.horizon
        x = self.norm_arr[start:mid, self.feature_cols]
        y = self.norm_arr[mid:end, 0]
        flags = self.raw_arr[mid:end, self.extreme_col].max()
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(flags, dtype=torch.int64),
        )


def pinball(preds, y, qs):
    losses = []
    for i, q in enumerate(qs):
        e = y - preds[..., i]
        losses.append(torch.max((q - 1) * e, q * e))
    return torch.mean(torch.stack(losses, dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=24 * 7)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_end = pd.to_datetime(cfg["splits"]["train_end"]).tz_localize("America/New_York")
    val_end = pd.to_datetime(cfg["splits"]["val_end"]).tz_localize("America/New_York")

    df = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "all.csv", parse_dates=["ds"]).sort_values("ds")
    feat_cols = ["y", "temp", "holiday", "extreme_flag"]
    extreme_col = feat_cols.index("extreme_flag")

    train_df = df[df["ds"] <= train_end]
    test_df = df[df["ds"] > val_end]

    train_vals = train_df[feat_cols].to_numpy()
    test_vals = test_df[feat_cols].to_numpy()

    mean = train_vals[:, :2].mean(axis=0)
    std = train_vals[:, :2].std(axis=0) + 1e-6

    def norm(v):
        v = v.copy()
        v[:, 0:2] = (v[:, 0:2] - mean) / std
        return v

    test_norm = norm(test_vals)

    ds = EvalWindowDataset(test_norm, test_vals, args.lookback, args.horizon, slice(0, len(feat_cols)), extreme_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model_cfg = LSTMConfig(
        input_size=len(feat_cols),
        lookback=args.lookback,
        horizon=args.horizon,
    )
    model = LSTMBaseline.load_from_checkpoint(args.checkpoint, cfg=model_cfg, map_location=device)
    model.to(device)
    model.eval(); model.freeze()

    all_pred = []
    all_true = []
    all_flag = []
    qs = list(model_cfg.quantiles)
    with torch.no_grad():
        for x, y, f in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            all_pred.append(preds.cpu())
            all_true.append(y.cpu())
            all_flag.append(f)

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    flags = torch.cat(all_flag, dim=0)

    median_pred = y_pred[..., 1]
    mae = torch.mean(torch.abs(median_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((median_pred - y_true) ** 2)).item()
    mape = torch.mean(torch.abs((median_pred - y_true) / (y_true + 1e-6))).item()
    pin = pinball(y_pred, y_true, qs).item()

    mask = flags == 1
    if mask.any():
        mae_ext = torch.mean(torch.abs(median_pred[mask] - y_true[mask])).item()
        rmse_ext = torch.sqrt(torch.mean((median_pred[mask] - y_true[mask]) ** 2)).item()
    else:
        mae_ext = rmse_ext = float("nan")

    print(f"Test MAE: {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAPE: {mape:.3f}")
    print(f"Test pinball: {pin:.3f}")
    print(f"Extreme MAE: {mae_ext:.3f}, RMSE: {rmse_ext:.3f}")


if __name__ == "__main__":
    main()
