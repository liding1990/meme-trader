"""Extract the 4-dimensional DNA fingerprint from GMGN token data.

DNA dimensions:
  1. Market Cap (from mcap candles)
  2. Holder count (from trends)
  3. Top 10 holder concentration (from trends)
  4. Market Cap / Holders ratio (derived from mcap + holder count)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from gmgn_api import fetch_token_data


def _safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_candle_series(candles_data):
    """Extract market cap and volume time series from mcap candle data.

    Returns a DataFrame with columns: timestamp, mcap, volume
    """
    candles = candles_data.get("data", {}).get("list", [])
    if not candles:
        return None

    rows = []
    for c in candles:
        t = int(c["time"])
        if t > 1e12:
            t = t // 1000  # ms → s
        rows.append({
            "timestamp": t,
            "mcap": _safe_float(c["close"]),
            "volume": _safe_float(c["volume"]),
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def extract_trend_series(trends_data):
    """Extract holder count and top10 concentration time series.

    Returns a DataFrame with columns: timestamp, holder_count, top10_pct
    """
    trends = trends_data.get("data", {}).get("trends", {})
    if not trends:
        return None

    holder_counts = trends.get("holder_count", [])
    top10 = trends.get("top10_holder_percent", [])

    if not holder_counts:
        return None

    # Build holder count series
    hc_data = {
        int(h["timestamp"]): int(_safe_float(h["value"]))
        for h in holder_counts
    }

    # Build top10 concentration series
    t10_data = {
        int(h["timestamp"]): _safe_float(h["value"]) * 100  # to percentage
        for h in top10
    } if top10 else {}

    # Merge on timestamps (use holder_count timestamps as base)
    all_ts = sorted(set(hc_data.keys()) | set(t10_data.keys()))

    rows = []
    for ts in all_ts:
        rows.append({
            "timestamp": ts,
            "holder_count": hc_data.get(ts, np.nan),
            "top10_pct": t10_data.get(ts, np.nan),
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df = df.interpolate(method="linear").dropna()
    return df


def build_dna_dataframe(candle_df, trend_df):
    """Merge candle and trend data into unified DNA DataFrame.

    The 4 DNA dimensions (aligned on hourly timestamps):
      1. mcap — market cap
      2. holder_count — number of unique holders
      3. top10_pct — top 10 holder supply concentration (%)
      4. mcap_per_holder — market cap / holder count

    Returns a DataFrame with columns:
        datetime, timestamp, mcap, holder_count, top10_pct, mcap_per_holder
    """
    if candle_df is None or trend_df is None:
        return None

    # Floor both sides to hour for alignment
    candle_df["hour"] = (candle_df["timestamp"] // 3600) * 3600
    trend_df["hour"] = (trend_df["timestamp"] // 3600) * 3600

    # Merge on hourly timestamp
    dna = pd.merge(
        candle_df[["hour", "mcap"]],
        trend_df[["hour", "holder_count", "top10_pct"]],
        on="hour",
        how="inner",
    )

    if dna.empty:
        return None

    dna["datetime"] = pd.to_datetime(dna["hour"], unit="s")
    dna["timestamp"] = dna["hour"]

    # Derived dimension: market cap per holder
    dna["mcap_per_holder"] = dna["mcap"] / dna["holder_count"].replace(0, np.nan)
    dna["mcap_per_holder"] = dna["mcap_per_holder"].fillna(0)

    dna = dna[["datetime", "timestamp", "mcap", "holder_count", "top10_pct", "mcap_per_holder"]]
    dna = dna.sort_values("timestamp").reset_index(drop=True)

    return dna


def fetch_and_extract_dna(chain, address):
    """End-to-end: fetch data from GMGN API and extract DNA fingerprint.

    Returns (data_dir, dna_df) where dna_df is the DNA DataFrame.
    """
    data_dir, loaded_data = fetch_token_data(chain, address)

    candle_df = extract_candle_series(loaded_data.get("token_mcap_candles", {}))
    trend_df = extract_trend_series(loaded_data.get("token_trends", {}))

    if candle_df is None:
        print("ERROR: No candle data available", file=sys.stderr)
        return data_dir, None
    if trend_df is None:
        print("ERROR: No trend data available", file=sys.stderr)
        return data_dir, None

    dna_df = build_dna_dataframe(candle_df, trend_df)

    if dna_df is not None:
        # Save DNA to CSV
        dna_path = os.path.join(data_dir, "dna.csv")
        dna_df.to_csv(dna_path, index=False)
        print(f"DNA fingerprint saved: {dna_path} ({len(dna_df)} data points)", file=sys.stderr)

        # Also save as JSON for inspection
        dna_json_path = os.path.join(data_dir, "dna.json")
        dna_df.to_json(dna_json_path, orient="records", indent=2, date_format="iso")
        print(f"DNA JSON saved: {dna_json_path}", file=sys.stderr)

    return data_dir, dna_df


def print_dna_summary(dna_df):
    """Print a summary of the DNA fingerprint."""
    if dna_df is None:
        print("No DNA data available.")
        return

    print(f"\n{'='*60}")
    print(f"  DNA Fingerprint Summary")
    print(f"{'='*60}")
    print(f"  Time range:    {dna_df['datetime'].iloc[0]} → {dna_df['datetime'].iloc[-1]}")
    print(f"  Data points:   {len(dna_df)} hours")
    print(f"{'─'*60}")
    print(f"  Market Cap:    {dna_df['mcap'].iloc[0]:,.2f} → {dna_df['mcap'].iloc[-1]:,.2f}")
    print(f"  Holders:       {dna_df['holder_count'].iloc[0]:,.0f} → {dna_df['holder_count'].iloc[-1]:,.0f}")
    print(f"  Top10 %:       {dna_df['top10_pct'].iloc[0]:.1f}% → {dna_df['top10_pct'].iloc[-1]:.1f}%")
    print(f"  MCap/Holder:   {dna_df['mcap_per_holder'].iloc[0]:,.2f} → {dna_df['mcap_per_holder'].iloc[-1]:,.2f}")
    print(f"{'='*60}\n")


def plot_dna(dna_df, save_path):
    """Plot the 4 DNA dimensions and save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if dna_df is None:
        print("No DNA data to plot.", file=sys.stderr)
        return

    dates = dna_df["datetime"]

    dims = [
        ("mcap", "Market Cap", "$"),
        ("holder_count", "Holders", ""),
        ("top10_pct", "Top 10 Holder %", "%"),
        ("mcap_per_holder", "MCap / Holder", "$"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Memecoin DNA Fingerprint", fontsize=14, fontweight="bold")

    for ax, (col, title, unit) in zip(axes, dims):
        ax.plot(dates, dna_df[col], linewidth=1.8, marker="o", markersize=4)
        ax.set_ylabel(f"{title} ({unit})" if unit else title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:00"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[-1].set_xlabel("Date", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {save_path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dna_extractor.py <contract_address> [chain]")
        print("  chain defaults to 'sol'")
        sys.exit(1)

    address = sys.argv[1]
    chain = sys.argv[2] if len(sys.argv) > 2 else "sol"

    data_dir, dna_df = fetch_and_extract_dna(chain, address)
    print_dna_summary(dna_df)

    if dna_df is not None:
        plot_path = os.path.join(data_dir, "dna_plot.png")
        plot_dna(dna_df, plot_path)
