import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.lstm_baseline import LSTMBaseline, LSTMConfig


class WindowDataset(Dataset):
    def __init__(self, arr: np.ndarray, lookback: int, horizon: int, feature_cols: slice):
        self.arr = arr
        self.lookback = lookback
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.n = arr.shape[0] - lookback - horizon + 1

    def __len__(self):
        return max(self.n, 0)

    def __getitem__(self, idx):
        start = idx
        mid = start + self.lookback
        end = mid + self.horizon
        x = self.arr[start:mid, self.feature_cols]
        y = self.arr[mid:end, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LoadDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, train_end: str, val_end: str, lookback: int, horizon: int, batch_size: int = 64, num_workers: int = 2):
        super().__init__()
        self.df = df.sort_values("ds")
        self.tz = self.df["ds"].dt.tz
        self.train_end = pd.to_datetime(train_end).tz_localize(self.tz)
        self.val_end = pd.to_datetime(val_end).tz_localize(self.tz)
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_df = self.df[self.df["ds"] <= self.train_end]
        val_df = self.df[(self.df["ds"] > self.train_end) & (self.df["ds"] <= self.val_end)]
        test_df = self.df[self.df["ds"] > self.val_end]

        feat_cols = ["y", "temp", "holiday", "extreme_flag"]
        self.feat_cols = feat_cols
        self.train_values = train_df[feat_cols].to_numpy()
        self.val_values = val_df[feat_cols].to_numpy()
        self.test_values = test_df[feat_cols].to_numpy()

        mean = self.train_values[:, :2].mean(axis=0)
        std = self.train_values[:, :2].std(axis=0) + 1e-6

        def norm(v):
            v = v.copy()
            v[:, 0:2] = (v[:, 0:2] - mean) / std
            return v

        self.train_values = norm(self.train_values)
        self.val_values = norm(self.val_values)
        self.test_values = norm(self.test_values)

        feat_slice = slice(0, len(feat_cols))
        self.train_ds = WindowDataset(self.train_values, self.lookback, self.horizon, feat_slice)
        self.val_ds = WindowDataset(self.val_values, self.lookback, self.horizon, feat_slice)
        self.test_ds = WindowDataset(self.test_values, self.lookback, self.horizon, feat_slice)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lookback", type=int, default=24 * 7)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_end = cfg["splits"]["train_end"]
    val_end = cfg["splits"]["val_end"]
    tz = cfg.get("timezone", "America/New_York")

    df = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "all.csv")
    df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce").dt.tz_convert(tz)
    df.dropna(subset=["ds"], inplace=True)

    dm = LoadDataModule(df, train_end, val_end, lookback=args.lookback, horizon=args.horizon, batch_size=args.batch_size)

    model_cfg = LSTMConfig(
        input_size=4,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lookback=args.lookback,
        horizon=args.horizon,
    )
    model = LSTMBaseline(model_cfg)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir="reports/lstm_baseline",
        log_every_n_steps=50,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()

