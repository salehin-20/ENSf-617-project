import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def make_datasets(df: pd.DataFrame, train_end, val_end, lookback: int, horizon: int):
    df = df.sort_values("ds").copy()
    df["time_idx"] = df["ds"].dt.tz_convert("UTC").view("int64") // 10**9 // 3600
    df["group"] = "nyiso"
    train_df = df[df["ds"] <= train_end]
    test_df = df[df["ds"] > val_end]

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
    test_ds = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    return training, test_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=24 * 14)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap for fast debug")
    parser.add_argument("--save_pred", type=Path, default=Path("reports/tft/preds.csv"))
    parser.add_argument("--save_metrics", type=Path, default=Path("reports/tft/metrics.yaml"))
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = yaml.safe_load(Path(args.config).read_text())
    quantiles = cfg.get("quantiles", [0.1, 0.5, 0.9])
    tz = cfg.get("timezone", "America/New_York")
    train_end = pd.to_datetime(cfg["splits"]["train_end"]).tz_localize(tz)
    val_end = pd.to_datetime(cfg["splits"]["val_end"]).tz_localize(tz)

    df = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "all.csv")
    df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce").dt.tz_convert(tz)
    df.dropna(subset=["ds"], inplace=True)

    _, test_ds = make_datasets(df, train_end, val_end, args.lookback, args.horizon)
    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    tft = TemporalFusionTransformer.load_from_checkpoint(args.checkpoint, map_location=device)
    tft.to(device)
    tft.eval()

    preds_all = []
    targets_all = []
    time_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            x, y = batch
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            y_t = y[0].to(device)  # already denormalized by target_normalizer
            out = tft(x)
            pred_denorm = out["prediction"]
            preds_all.append(pred_denorm.cpu())
            targets_all.append(y_t.cpu())
            time_all.append(x["decoder_time_idx"].cpu())

    preds = torch.cat(preds_all, dim=0)  # (N, horizon, Q)
    targets = torch.cat(targets_all, dim=0)  # (N, horizon)
    times = torch.cat(time_all, dim=0)  # (N, horizon)

    median_idx = len(quantiles) // 2
    median_preds = preds[..., median_idx].reshape(-1)
    targets_flat = targets.reshape(-1)

    mae = torch.mean(torch.abs(median_preds - targets_flat)).item()
    rmse = torch.sqrt(torch.mean((median_preds - targets_flat) ** 2)).item()

    denom = torch.clamp(targets_flat.abs(), min=1000.0)
    mape = torch.mean(torch.abs((median_preds - targets_flat) / denom)).item() * 100
    qloss = QuantileLoss(quantiles)(preds, targets).item()

    print(f"Test MAE: {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Test pinball: {qloss:.3f}")

    args.save_pred.parent.mkdir(parents=True, exist_ok=True)
    args.save_metrics.parent.mkdir(parents=True, exist_ok=True)

    # convert time_idx (hours since epoch) back to timezone-aware timestamps
    ds_series = pd.to_datetime(times.reshape(-1).numpy() * 3600, unit="s", utc=True).tz_convert(tz)

    flat_preds = preds.reshape(-1, len(quantiles)).numpy()
    target_vals = targets_flat.numpy()
    pred_df = pd.DataFrame({"ds": ds_series, "y": target_vals})
    for qi, q in enumerate(quantiles):
        pred_df[f"p{int(q*100):02d}"] = flat_preds[:, qi]
    pred_df.to_csv(args.save_pred, index=False)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "pinball": qloss,
        "samples": len(pred_df),
    }
    args.save_metrics.write_text(yaml.safe_dump(metrics))


if __name__ == "__main__":
    main()


