import argparse
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return pd.read_csv(path, parse_dates=["ds"])


def plot_calibration(pred_df: pd.DataFrame, quantiles, out_path: Path):
    rows = []
    y = pred_df["y"].values
    for q in quantiles:
        col = f"p{int(q*100):02d}"
        if col not in pred_df:
            continue
        cover = (y <= pred_df[col].values).mean()
        rows.append((q, cover))
    qs, cov = zip(*rows)
    plt.figure(figsize=(4, 4))
    plt.plot(qs, qs, "k--", label="ideal")
    plt.plot(qs, cov, "o-", label="TFT")
    plt.xlabel("Nominal quantile")
    plt.ylabel("Empirical coverage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sample_day(pred_df: pd.DataFrame, day: pd.Timestamp, out_path: Path):
    day_df = pred_df[pred_df["ds"].dt.date == day.date()].copy()
    if day_df.empty:
        return
    day_df = day_df.sort_values("ds")
    plt.figure(figsize=(9, 3.5))
    plt.plot(day_df["ds"], day_df["y"], label="Actual", color="black")
    if "p50" in day_df:
        plt.plot(day_df["ds"], day_df["p50"], label="Median", color="C0")
    if {"p10", "p90"}.issubset(day_df.columns):
        plt.fill_between(day_df["ds"], day_df["p10"], day_df["p90"], color="C0", alpha=0.2, label="P10-P90")
    plt.title(f"Forecast for {day.date()}")
    plt.ylabel("Load (MW)")
    plt.xticks(rotation=25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_extremes(pred_df: pd.DataFrame, data_df: pd.DataFrame | None, out_path: Path):
    if data_df is not None and "extreme_flag" in data_df.columns:
        merged = pred_df.merge(data_df[["ds", "extreme_flag"]], on="ds", how="left")
    else:
        merged = pred_df.copy()
        q5 = merged["y"].quantile(0.05)
        q95 = merged["y"].quantile(0.95)
        merged["extreme_flag"] = ((merged["y"] <= q5) | (merged["y"] >= q95)).astype(int)
    merged["abs_err"] = (merged["p50"] - merged["y"]).abs()
    bars = merged.groupby("extreme_flag")["abs_err"].mean()
    plt.figure(figsize=(3.5, 3.5))
    bars.plot(kind="bar", color=["C1", "C0"])
    plt.xticks([0, 1], ["Normal", "Extreme"], rotation=0)
    plt.ylabel("MAE (MW)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--pred_path", default="reports/tft/preds.csv")
    parser.add_argument("--out_dir", default="reports")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    quantiles = cfg.get("quantiles", [0.1, 0.5, 0.9])

    tz = cfg.get("timezone", "America/New_York")
    pred_df = load_predictions(Path(args.pred_path))
    pred_df["ds"] = pd.to_datetime(pred_df["ds"], utc=True, errors="coerce").dt.tz_convert(tz)
    pred_df.dropna(subset=["ds"], inplace=True)

    data_path = Path(cfg["data"]["processed_dir"]) / "all.csv"
    data_df = None
    if data_path.exists():
        data_df = pd.read_csv(data_path)
        data_df["ds"] = pd.to_datetime(data_df["ds"], utc=True, errors="coerce").dt.tz_convert(tz)
        data_df.dropna(subset=["ds"], inplace=True)

    out_dir = Path(args.out_dir)
    plot_calibration(pred_df, quantiles, out_dir / "calibration_tft.png")

    last_day = pred_df["ds"].max().normalize()
    plot_sample_day(pred_df, last_day, out_dir / "sample_day_tft.png")

    plot_extremes(pred_df, data_df, out_dir / "extreme_mae_tft.png")

if __name__ == "__main__":
    main()
