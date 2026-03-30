"""Survival analysis for momentum exit timing.

Models the "lifetime" of momentum episodes: given a pump has lasted T bars
with certain characteristics, what's the probability it dies in the next bar?

Two models:
1. Kaplan-Meier: non-parametric baseline — raw survival curves
2. Cox Proportional Hazards: semi-parametric — hazard rate as function of covariates

Covariates for Cox PH:
  - initial_roc: rate of change at start of momentum episode
  - peak_roc: maximum ROC during the episode
  - rvol: average relative volume during episode
  - mcap_at_entry: market cap when momentum started
  - holder_growth: hourly holder growth rate (from hourly data)

Usage:
    python survival.py train      # train survival models on historical data
    python survival.py analyze    # show survival curves and hazard rates
"""

import argparse
import json
import glob
import os
import sys
import pickle

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter

from indicators import extract_5m_candles, compute_all_indicators
from space4d import load_token_list, MCAP_THRESHOLD

DATA_DIR = "data"
PLOTS_DIR = "plots"


def detect_pump_episodes(df, min_roc=5.0, min_duration=3):
    """Detect pump episodes from 5-min candle data.

    A pump episode starts when ROC exceeds min_roc% and ends when
    mcap drops below the entry level OR ROC turns significantly negative.

    Returns list of episode dicts.
    """
    if "roc_30m" not in df.columns:
        df = compute_all_indicators(df)

    episodes = []
    in_pump = False
    entry_idx = None
    entry_mcap = 0
    peak_mcap = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        roc = row.get("roc_30m", 0)

        if pd.isna(roc):
            continue

        if not in_pump:
            if roc > min_roc:
                in_pump = True
                entry_idx = idx
                entry_mcap = row["mcap"]
                peak_mcap = row["mcap"]
        else:
            current_mcap = row["mcap"]
            if current_mcap > peak_mcap:
                peak_mcap = current_mcap

            # Pump ends when:
            # 1. Price drops below entry
            # 2. Significant drawdown from peak (>20%)
            drawdown = (1 - current_mcap / peak_mcap) * 100 if peak_mcap > 0 else 0
            below_entry = current_mcap < entry_mcap

            if below_entry or drawdown > 20:
                duration = idx - entry_idx
                if duration >= min_duration:
                    # Compute episode features
                    episode_df = df.iloc[entry_idx:idx + 1]
                    peak_return = (peak_mcap / entry_mcap - 1) * 100 if entry_mcap > 0 else 0

                    episodes.append({
                        "entry_idx": entry_idx,
                        "exit_idx": idx,
                        "duration": duration,  # in 5-min bars
                        "entry_mcap": entry_mcap,
                        "peak_mcap": peak_mcap,
                        "exit_mcap": current_mcap,
                        "peak_return_pct": peak_return,
                        "exit_return_pct": (current_mcap / entry_mcap - 1) * 100,
                        "initial_roc": df.iloc[entry_idx].get("roc_30m", 0),
                        "avg_rvol": episode_df["rvol"].mean() if "rvol" in episode_df.columns else 1.0,
                        "mcap_at_entry": entry_mcap,
                        "observed": True,  # episode completed (not censored)
                    })

                in_pump = False
                entry_idx = None

    # Handle ongoing pump at end of data (censored observation)
    if in_pump and entry_idx is not None:
        duration = len(df) - 1 - entry_idx
        if duration >= min_duration:
            episode_df = df.iloc[entry_idx:]
            peak_return = (peak_mcap / entry_mcap - 1) * 100 if entry_mcap > 0 else 0
            episodes.append({
                "entry_idx": entry_idx,
                "exit_idx": len(df) - 1,
                "duration": duration,
                "entry_mcap": entry_mcap,
                "peak_mcap": peak_mcap,
                "exit_mcap": df.iloc[-1]["mcap"],
                "peak_return_pct": peak_return,
                "exit_return_pct": (df.iloc[-1]["mcap"] / entry_mcap - 1) * 100,
                "initial_roc": df.iloc[entry_idx].get("roc_30m", 0),
                "avg_rvol": episode_df["rvol"].mean() if "rvol" in episode_df.columns else 1.0,
                "mcap_at_entry": entry_mcap,
                "observed": False,  # censored — pump still ongoing at end of data
            })

    return episodes


def build_survival_dataset(tokens):
    """Build a survival analysis dataset from all tokens' pump episodes."""
    all_episodes = []

    for token in tokens:
        address = token["address"]
        df = extract_5m_candles(address)
        if df is None or len(df) < 20:
            continue

        episodes = detect_pump_episodes(df)
        for ep in episodes:
            ep["symbol"] = token["symbol"]
            ep["address"] = address
        all_episodes.extend(episodes)

    if not all_episodes:
        return None

    return pd.DataFrame(all_episodes)


def train_models(episodes_df):
    """Train Kaplan-Meier and Cox PH models."""
    # Kaplan-Meier (non-parametric baseline)
    print("Training Kaplan-Meier...")
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=episodes_df["duration"],
        event_observed=episodes_df["observed"],
        label="Momentum Episode Survival",
    )

    median_survival = kmf.median_survival_time_
    print(f"  Median survival time: {median_survival:.0f} bars ({median_survival * 5:.0f} minutes)")
    print(f"  Survival at 6 bars (30 min): {kmf.predict(6):.1%}")
    print(f"  Survival at 12 bars (1 hr):  {kmf.predict(12):.1%}")
    print(f"  Survival at 36 bars (3 hr):  {kmf.predict(36):.1%}")

    # Cox PH (semi-parametric with covariates)
    print("\nTraining Cox Proportional Hazards...")
    cox_features = ["duration", "observed", "initial_roc", "avg_rvol", "peak_return_pct"]
    cox_df = episodes_df[cox_features].copy()
    cox_df["log_mcap"] = np.log1p(episodes_df["mcap_at_entry"])

    # Drop rows with NaN/inf
    cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(cox_df) < 10:
        print("  Not enough data for Cox PH")
        return kmf, None

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(
            cox_df,
            duration_col="duration",
            event_col="observed",
        )
        print(cph.print_summary())
    except Exception as e:
        print(f"  Cox PH failed: {e}")
        cph = None

    return kmf, cph


def compute_hazard_rate(kmf, current_duration):
    """Compute the hazard rate at a given duration.

    Returns P(episode ends in next bar | survived to current_duration).
    """
    survival = kmf.predict(current_duration)
    survival_next = kmf.predict(current_duration + 1)

    if survival > 0:
        hazard = 1 - (survival_next / survival)
    else:
        hazard = 1.0

    return float(np.clip(hazard, 0, 1))


def save_models(kmf, cph):
    os.makedirs("models", exist_ok=True)
    with open("models/survival.pkl", "wb") as f:
        pickle.dump({"kmf": kmf, "cph": cph}, f)
    print("Saved: models/survival.pkl")


def load_models():
    with open("models/survival.pkl", "rb") as f:
        d = pickle.load(f)
    return d["kmf"], d["cph"]


def plot_survival(kmf, episodes_df):
    """Plot survival curve and hazard function."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Kaplan-Meier Survival Curve",
            "Hazard Rate (P(dies next bar | survived to t))",
            "Episode Duration Distribution",
        ],
        vertical_spacing=0.1,
    )

    # Survival curve
    timeline = np.arange(0, episodes_df["duration"].max() + 1)
    survival = [float(kmf.predict(t)) for t in timeline]
    fig.add_trace(go.Scatter(
        x=timeline * 5,  # convert bars to minutes
        y=survival,
        mode="lines",
        line=dict(color="#1f77b4", width=2),
        name="Survival Probability",
    ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=1)

    # Hazard rate
    hazards = [compute_hazard_rate(kmf, t) for t in timeline]
    fig.add_trace(go.Scatter(
        x=timeline * 5,
        y=hazards,
        mode="lines",
        line=dict(color="#d62728", width=2),
        name="Hazard Rate",
    ), row=2, col=1)

    # Duration histogram
    fig.add_trace(go.Histogram(
        x=episodes_df["duration"] * 5,
        nbinsx=30,
        marker_color="#2ca02c",
        name="Episode Count",
    ), row=3, col=1)

    fig.update_layout(
        title=f"Momentum Episode Survival Analysis ({len(episodes_df)} episodes)",
        height=900, width=1000,
    )
    fig.update_xaxes(title_text="Minutes", row=3, col=1)
    fig.update_yaxes(title_text="P(survive)", row=1, col=1)
    fig.update_yaxes(title_text="P(die next bar)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    path = os.path.join(PLOTS_DIR, "survival_analysis.html")
    fig.write_html(path)
    print(f"Plot saved: {path}")
    return path


def cmd_train(args):
    tokens = load_token_list()
    print(f"Building survival dataset from {len(tokens)} tokens...")

    episodes_df = build_survival_dataset(tokens)
    if episodes_df is None or len(episodes_df) < 10:
        print("ERROR: Not enough pump episodes found. Need 5m data — run fetch_5m.py first.")
        sys.exit(1)

    print(f"Found {len(episodes_df)} pump episodes")
    print(f"  Observed (completed): {episodes_df['observed'].sum()}")
    print(f"  Censored (ongoing):   {(~episodes_df['observed']).sum()}")
    print(f"  Duration range: {episodes_df['duration'].min()}-{episodes_df['duration'].max()} bars "
          f"({episodes_df['duration'].min()*5}-{episodes_df['duration'].max()*5} min)")
    print(f"  Median peak return: {episodes_df['peak_return_pct'].median():+.1f}%")

    kmf, cph = train_models(episodes_df)
    save_models(kmf, cph)

    # Save episodes for analysis
    episodes_df.to_csv(os.path.join(DATA_DIR, "pump_episodes.csv"), index=False)

    # Plot
    path = plot_survival(kmf, episodes_df)
    import subprocess
    subprocess.run(["open", path])


def cmd_analyze(args):
    kmf, cph = load_models()

    print("\nHazard rate at different momentum durations:")
    for bars in [3, 6, 12, 18, 24, 36, 48, 72]:
        hz = compute_hazard_rate(kmf, bars)
        print(f"  {bars} bars ({bars*5:>3d} min): hazard = {hz:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Survival Analysis for Momentum Exit")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train")
    sub.add_parser("analyze")
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
