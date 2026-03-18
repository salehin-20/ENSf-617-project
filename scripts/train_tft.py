import argparse
import yaml
import pandas as pd
from pathlib import Path
import sys
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def make_datasets(df: pd.DataFrame, train_end, val_end, lookback: int, horizon: int):
    df = df.sort_values("ds").copy()
    df["time_idx"] = df["ds"].astype("int64") // 10**9 // 3600
    df["group"] = "nyiso"
    train_df = df[df["ds"] <= train_end]
    val_df = df[(df["ds"] > train_end) & (df["ds"] <= val_end)]

    max_encoder_length = lookback
    max_prediction_length = horizon

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="y",
        group_ids=["group"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group"],
        time_varying_known_reals=["holiday"],
        time_varying_unknown_reals=["y", "temp", "extreme_flag"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        target_normalizer=GroupNormalizer(groups=["group"]),
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
    return training, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lookback", type=int, default=24*14)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    quantiles = cfg.get("quantiles", [0.1, 0.5, 0.9])
    train_end = pd.to_datetime(cfg["splits"]["train_end"]).tz_localize("America/New_York")
    val_end = pd.to_datetime(cfg["splits"]["val_end"]).tz_localize("America/New_York")

    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "all.parquet")

    train_ds, val_ds = make_datasets(df, train_end, val_end, args.lookback, args.horizon)

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=1e-3,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_heads,
        dropout=args.dropout,
        loss=QuantileLoss(quantiles=quantiles),
        output_size=len(quantiles),
        log_interval=50,
        reduce_on_plateau_patience=3,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir="reports/tft",
        log_every_n_steps=50,
    )

    train_loader = train_ds.to_dataloader(train=True, batch_size=args.batch_size, num_workers=2)
    val_loader = val_ds.to_dataloader(train=False, batch_size=args.batch_size, num_workers=2)

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
