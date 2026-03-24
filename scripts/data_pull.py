"""Data pull for NYISO load + Open-Meteo weather with EIA v2 primary and MIS fallback.
"""
import argparse
import datetime as dt
import os
from pathlib import Path
import sys
import yaml
import requests
import pandas as pd
import holidays


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def download_nyiso_day(day: dt.date, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"rtload_{day:%Y%m%d}.csv"
    if fname.exists():
        return fname
    url = f"http://mis.nyiso.com/public/csv/rtload/rtload_{day:%Y%m%d}.csv"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"NYISO download failed {resp.status_code} for {day}")
    fname.write_bytes(resp.content)
    return fname


def fetch_load_mis(start: dt.date, end: dt.date, raw_dir: Path, tz: str) -> pd.DataFrame:
    csv_files = []
    for day in daterange(start, end):
        try:
            csv_files.append(download_nyiso_day(day, raw_dir))
        except Exception as exc:
            print(f"[warn] skip {day}: {exc}", file=sys.stderr)
    records = []
    for path in csv_files:
        df = pd.read_csv(path)
        if not {"Time Stamp", "Name", "Load"}.issubset(df.columns):
            print(f"[warn] unexpected columns in {path.name}", file=sys.stderr)
            continue
        nyca = df[df["Name"] == "NYCA"].copy()
        nyca.rename(columns={"Time Stamp": "ds", "Load": "y"}, inplace=True)
        records.append(nyca[["ds", "y"]])
    if not records:
        raise RuntimeError("No NYISO load data downloaded")
    load = pd.concat(records, ignore_index=True)
    load["ds"] = pd.to_datetime(load["ds"], errors="coerce")
    load.dropna(subset=["ds", "y"], inplace=True)
    load["ds"] = load["ds"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    load = load.groupby("ds", as_index=False)["y"].mean()
    return load


def fetch_load_eia(start: dt.date, end: dt.date, tz: str) -> pd.DataFrame:
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("EIA_API_KEY not set")
    series_id = "EBA.NYIS-ALL.D.H"
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    all_rows = []
    offset = 0
    page = 0
    while True:
        params = {
            "api_key": api_key,
            "start": f"{start}T00",
            "end": f"{end}T23",
            "offset": offset,
            "length": 5000,
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"EIA v2 request failed {resp.status_code}: {resp.text[:200]}")
        payload = resp.json().get("response", {})
        data = payload.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        print(f"EIA page {page} fetched {len(data)} rows", file=sys.stderr)
        if len(data) < 5000:
            break
        offset += len(data)
        page += 1
    if not all_rows:
        raise RuntimeError("EIA returned no data")
    df = pd.DataFrame(all_rows)
    if "value" not in df.columns or "period" not in df.columns:
        raise RuntimeError("Unexpected EIA schema")
    df.rename(columns={"value": "y", "period": "ds"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_convert(tz)
    df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    return df


def fetch_weather(start: dt.date, end: dt.date, raw_dir: Path, tz: str) -> pd.DataFrame:
    ensure_dirs(raw_dir)
    lat, lon = 40.7128, -74.0060
    frames = []
    cur = start
    step = dt.timedelta(days=365)  # 1-year chunks
    while cur <= end:
        chunk_end = min(cur + step, end)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": cur.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ["temperature_2m", "dewpoint_2m", "precipitation"],
            "timezone": tz,
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"[warn] open-meteo {cur}..{chunk_end} failed {resp.status_code}", file=sys.stderr)
            cur = chunk_end + dt.timedelta(days=1)
            continue
        data = resp.json().get("hourly") or {}
        times = data.get("time") or []
        temps = data.get("temperature_2m") or []
        dewps = data.get("dewpoint_2m") or []
        prcps = data.get("precipitation") or []
        if not times:
            cur = chunk_end + dt.timedelta(days=1)
            continue
        df = pd.DataFrame({"ds": times, "temp": temps, "dwpt": dewps, "prcp": prcps})
        df["ds"] = pd.to_datetime(df["ds"], utc=False, errors="coerce")
        df.dropna(subset=["ds"], inplace=True)
        df["ds"] = df["ds"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        df.dropna(subset=["ds"], inplace=True)
        frames.append(df)
        cur = chunk_end + dt.timedelta(days=1)
    if not frames:
        raise RuntimeError("No weather returned from open-meteo")
    wx = pd.concat(frames, ignore_index=True)
    wx.dropna(subset=["ds"], inplace=True)
    wx = wx.drop_duplicates(subset=["ds"]).sort_values("ds")
    wx.to_csv(raw_dir / "weather_raw.csv", index=False)
    return wx


def add_features(df: pd.DataFrame, tz: str, extreme_cfg: dict) -> pd.DataFrame:
    us_holidays = holidays.US()
    df["holiday"] = df["ds"].dt.tz_convert(tz).dt.date.map(lambda d: 1 if d in us_holidays else 0)
    low_pct = extreme_cfg.get("low_pct", 5)
    high_pct = extreme_cfg.get("high_pct", 95)
    low_thr = df["temp"].quantile(low_pct / 100)
    high_thr = df["temp"].quantile(high_pct / 100)
    df["extreme_flag"] = ((df["temp"] <= low_thr) | (df["temp"] >= high_thr)).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Pull NYISO load + weather")
    parser.add_argument("--config", default="src/config.yaml", help="Path to config YAML")
    parser.add_argument("--refresh-weather", action="store_true", help="Force re-download weather archive")
    parser.add_argument("--force", action="store_true", help="Recompute processed file even if it exists")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    tz = os.environ.get("TZ", cfg.get("timezone", "America/New_York"))
    span = cfg.get("span", {})
    start = pd.to_datetime(span.get("start")).date()
    end = span.get("end")
    end = pd.to_datetime(end).date() if end else dt.date.today()

    raw_load_dir = Path(cfg["data"]["raw_dir"]) / "load"
    raw_weather_dir = Path(cfg["data"]["raw_dir"]) / "weather"
    processed_dir = Path(cfg["data"]["processed_dir"])
    ensure_dirs(raw_load_dir, raw_weather_dir, processed_dir)
    out_path = processed_dir / "all.csv"

    if out_path.exists() and not args.force:
        print(f"{out_path} already exists; use --force to rebuild")
        return

    print(f"Downloading NYISO load {start} to {end}...")
    api_key_set = os.environ.get("EIA_API_KEY") is not None
    if api_key_set:
        load_df = fetch_load_eia(start, end, tz)
        print("Loaded from EIA API v2 (EBA.NYIS-ALL.D.H)")
    else:
        print("[warn] EIA_API_KEY not set; using short-window MIS CSVs")
        load_df = fetch_load_mis(start, end, raw_load_dir, tz)

    print(f"Fetching Open-Meteo weather {start} to {end}...")
    cached_weather = raw_weather_dir / "weather_raw.csv"
    if cached_weather.exists() and not args.refresh_weather:
        weather_df = pd.read_csv(cached_weather)
        weather_df["ds"] = pd.to_datetime(weather_df["ds"], utc=True, errors="coerce")
        weather_df.dropna(subset=["ds"], inplace=True)
        weather_df["ds"] = weather_df["ds"].dt.tz_convert(tz)
        print(f"Loaded cached weather from {cached_weather}")
    else:
        weather_df = fetch_weather(start, end, raw_weather_dir, tz)

    print("Merging load + weather on ds hour...")
    merged = pd.merge_asof(
        load_df.sort_values("ds"),
        weather_df.sort_values("ds"),
        on="ds",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )
    merged.dropna(subset=["y", "temp"], inplace=True)
    merged = add_features(merged, tz, cfg.get("extreme", {}))

    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()
