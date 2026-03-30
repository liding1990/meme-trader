"""Kalman Filter + CUSUM momentum detection engine.

Transforms raw 4D token data into stationary features, tracks momentum
via Kalman filter (local linear trend model), detects regime changes
via CUSUM.

Features (computed per hour):
  x1: Δlog(mcap)              — market cap log return
  x2: Δholders / holders      — holder growth rate
  x3: Δtop10_pct              — concentration change

Kalman state (6-dim):
  [level_x1, trend_x1, level_x2, trend_x2, level_x3, trend_x3]
  The trend components are the smoothed momentum signals.

Entry: trend_mcap > 0 AND trend_holders > 0
Exit:  CUSUM detects downward shift in trend_mcap

Usage:
    python momentum.py analyze <contract_address> [--chain sol]
    python momentum.py scan  # scan all tokens for active momentum
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from pykalman import KalmanFilter

from space4d import extract_trajectory, load_token_list, MCAP_THRESHOLD

DATA_DIR = "data"
PLOTS_DIR = "plots"


def compute_features(trajectory):
    """Transform raw (T, 3) trajectory into (T-1, 3) stationary features.

    Input columns: [mcap, holders, top10_pct]
    Output columns: [log_return_mcap, holder_growth_rate, delta_top10]
    """
    mcap = trajectory[:, 0]
    holders = trajectory[:, 1]
    top10 = trajectory[:, 2]

    # Guard against zero/negative values
    mcap_safe = np.clip(mcap, 1.0, None)
    holders_safe = np.clip(holders, 1.0, None)

    # Stationary features
    log_return = np.diff(np.log(mcap_safe))
    holder_growth = np.diff(holders_safe) / holders_safe[:-1]
    delta_top10 = np.diff(top10)

    features = np.column_stack([log_return, holder_growth, delta_top10])

    # Clip extreme outliers (> 5 std)
    for col in range(features.shape[1]):
        std = np.std(features[:, col])
        if std > 0:
            features[:, col] = np.clip(features[:, col], -5 * std, 5 * std)

    return features


def build_kalman_filter(n_features=3, em_iters=5):
    """Build a local linear trend Kalman filter.

    State: [level_1, trend_1, level_2, trend_2, level_3, trend_3]
    Each feature has a level + trend component.
    """
    n_state = n_features * 2

    # Transition matrix: block diagonal [[1, 1], [0, 1]] per feature
    F = np.zeros((n_state, n_state))
    for i in range(n_features):
        F[2 * i, 2 * i] = 1.0       # level -> level
        F[2 * i, 2 * i + 1] = 1.0   # trend -> level
        F[2 * i + 1, 2 * i + 1] = 1.0  # trend -> trend

    # Observation matrix: observe only the level components
    H = np.zeros((n_features, n_state))
    for i in range(n_features):
        H[i, 2 * i] = 1.0

    # Initial state and covariance
    initial_state = np.zeros(n_state)
    initial_cov = np.eye(n_state) * 1.0

    # Default process and observation noise
    Q = np.eye(n_state) * 0.01
    R = np.eye(n_features) * 0.1

    kf = KalmanFilter(
        transition_matrices=F,
        observation_matrices=H,
        initial_state_mean=initial_state,
        initial_state_covariance=initial_cov,
        transition_covariance=Q,
        observation_covariance=R,
        em_vars=["transition_covariance", "observation_covariance"],
    )

    return kf


def run_kalman(features, em_iters=5):
    """Fit Kalman filter and return filtered state means.

    Returns (T, 6) state means where:
      col 0: level_mcap_return
      col 1: trend_mcap_return  ← primary momentum signal
      col 2: level_holder_growth
      col 3: trend_holder_growth  ← secondary momentum signal
      col 4: level_top10_change
      col 5: trend_top10_change
    """
    kf = build_kalman_filter(n_features=features.shape[1], em_iters=em_iters)

    # EM parameter estimation (if enough data)
    if len(features) >= 20:
        try:
            kf = kf.em(features, n_iter=em_iters)
        except Exception:
            pass  # fall back to defaults

    # Online filtering (no look-ahead)
    state_means, state_covs = kf.filter(features)

    return state_means


def cusum_detect(signal, h=1.0, d=0.0):
    """CUSUM change detection on a 1D signal.

    Returns arrays of (cusum_high, cusum_low, exit_signal).
    exit_signal[t] = True when downward shift detected.

    h: threshold (in units of signal std)
    d: drift allowance
    """
    T = len(signal)
    sig_std = np.std(signal)
    if sig_std <= 0:
        sig_std = 1.0

    threshold = h * sig_std

    s_high = np.zeros(T)
    s_low = np.zeros(T)
    exit_signal = np.zeros(T, dtype=bool)

    for t in range(1, T):
        s_high[t] = max(0, s_high[t - 1] + signal[t] - d * sig_std)
        s_low[t] = min(0, s_low[t - 1] + signal[t] + d * sig_std)

        # Detect downward shift (momentum dying)
        if s_low[t] < -threshold:
            exit_signal[t] = True
            s_low[t] = 0  # reset after detection

        # Reset upward accumulator if it triggers (optional: track both)
        if s_high[t] > threshold:
            s_high[t] = 0

    return s_high, s_low, exit_signal


def generate_signals(trajectory, h=1.0, d=0.0, em_iters=5):
    """Full pipeline: raw trajectory → features → Kalman → CUSUM → entry/exit signals.

    Returns DataFrame with all intermediate values and signals.
    """
    if len(trajectory) < 5:
        return None

    features = compute_features(trajectory)
    if len(features) < 3:
        return None

    state_means = run_kalman(features, em_iters=em_iters)

    trend_mcap = state_means[:, 1]
    trend_holders = state_means[:, 3]
    trend_top10 = state_means[:, 5]

    # CUSUM on mcap trend
    cusum_high, cusum_low, cusum_exit = cusum_detect(trend_mcap, h=h, d=d)

    # Entry: both mcap and holder trends positive
    entry = (trend_mcap > 0) & (trend_holders > 0)

    # Exit: CUSUM downward detection OR both trends negative (backstop)
    both_negative = (trend_mcap < 0) & (trend_holders < 0)
    exit_signal = cusum_exit | both_negative

    # Build DataFrame (aligned to features, which start at hour 1)
    T = len(features)
    df = pd.DataFrame({
        "hour": np.arange(1, T + 1),
        "mcap": trajectory[1:T + 1, 0],
        "holders": trajectory[1:T + 1, 1],
        "top10_pct": trajectory[1:T + 1, 2],
        "x1_log_return": features[:, 0],
        "x2_holder_growth": features[:, 1],
        "x3_delta_top10": features[:, 2],
        "trend_mcap": trend_mcap,
        "trend_holders": trend_holders,
        "trend_top10": trend_top10,
        "cusum_high": cusum_high,
        "cusum_low": cusum_low,
        "entry": entry,
        "exit": exit_signal,
    })

    return df


def plot_analysis(df, symbol, address):
    """Plot momentum analysis for a single token."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Market Cap ($)",
            "Kalman Trends (MCap & Holders)",
            "CUSUM Statistic",
            "Holders",
            "Top10 Holder %",
        ],
        vertical_spacing=0.05,
    )

    hours = df["hour"]

    # Row 1: MCap with entry/exit markers
    fig.add_trace(go.Scatter(
        x=hours, y=df["mcap"], mode="lines",
        line=dict(color="black", width=2), name="MCap",
    ), row=1, col=1)

    entries = df[df["entry"] & ~df["entry"].shift(1, fill_value=False)]
    exits = df[df["exit"]]
    fig.add_trace(go.Scatter(
        x=entries["hour"], y=entries["mcap"], mode="markers",
        marker=dict(color="green", size=10, symbol="triangle-up"),
        name="Entry Signal",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=exits["hour"], y=exits["mcap"], mode="markers",
        marker=dict(color="red", size=10, symbol="triangle-down"),
        name="Exit Signal",
    ), row=1, col=1)

    # Row 2: Kalman trends
    fig.add_trace(go.Scatter(
        x=hours, y=df["trend_mcap"], mode="lines",
        line=dict(color="#1f77b4", width=2), name="MCap Trend",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=hours, y=df["trend_holders"], mode="lines",
        line=dict(color="#ff7f0e", width=2), name="Holder Trend",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Row 3: CUSUM
    fig.add_trace(go.Scatter(
        x=hours, y=df["cusum_high"], mode="lines",
        line=dict(color="green", width=1.5), name="CUSUM High",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=hours, y=df["cusum_low"], mode="lines",
        line=dict(color="red", width=1.5), name="CUSUM Low",
    ), row=3, col=1)

    # Row 4: Holders
    fig.add_trace(go.Scatter(
        x=hours, y=df["holders"], mode="lines",
        line=dict(color="#2ca02c", width=2), name="Holders",
    ), row=4, col=1)

    # Row 5: Top10%
    fig.add_trace(go.Scatter(
        x=hours, y=df["top10_pct"], mode="lines",
        line=dict(color="#9467bd", width=2), name="Top10%",
    ), row=5, col=1)

    fig.update_layout(
        title=f"Momentum Analysis: {symbol} ({address[:8]})",
        height=1200, width=1200,
    )
    fig.update_xaxes(title_text="Hours from 50K", row=5, col=1)

    path = os.path.join(PLOTS_DIR, f"momentum_{symbol}.html")
    fig.write_html(path)
    print(f"Plot saved: {path}")
    return path


def cmd_analyze(args):
    """Analyze a single token's momentum."""
    from gmgn_api import fetch_token_data
    from dna_extractor import extract_candle_series, extract_trend_series, build_dna_dataframe

    address = args.address
    chain = args.chain

    # Try loading from cache first, fetch if not available
    traj = extract_trajectory(address)
    if traj is None:
        print(f"Fetching {address[:8]}...")
        _, loaded_data = fetch_token_data(chain, address)
        candle_df = extract_candle_series(loaded_data.get("token_mcap_candles", {}))
        trend_df = extract_trend_series(loaded_data.get("token_trends", {}))
        if candle_df is None or trend_df is None:
            print("ERROR: Could not fetch data", file=sys.stderr)
            return
        dna_df = build_dna_dataframe(candle_df, trend_df)
        if dna_df is None:
            print("ERROR: Could not build DNA", file=sys.stderr)
            return
        mcap = dna_df["mcap"].values
        above = np.where(mcap >= MCAP_THRESHOLD)[0]
        if len(above) == 0:
            print(f"ERROR: Never reached ${MCAP_THRESHOLD:,}", file=sys.stderr)
            return
        trimmed = dna_df.iloc[above[0]:].reset_index(drop=True)
        traj = trimmed[["mcap", "holder_count", "top10_pct"]].values

    print(f"Trajectory: {len(traj)} hours")

    df = generate_signals(traj, h=args.h, d=args.d)
    if df is None:
        print("ERROR: Not enough data for analysis", file=sys.stderr)
        return

    # Summary
    n_entries = (df["entry"] & ~df["entry"].shift(1, fill_value=False)).sum()
    n_exits = df["exit"].sum()
    current_trend_mcap = df["trend_mcap"].iloc[-1]
    current_trend_holders = df["trend_holders"].iloc[-1]
    in_momentum = current_trend_mcap > 0 and current_trend_holders > 0

    print(f"\n{'='*60}")
    print(f"  Momentum Analysis: {address[:8]}")
    print(f"{'='*60}")
    print(f"  Current MCap:       ${df['mcap'].iloc[-1]:,.0f}")
    print(f"  Current Holders:    {df['holders'].iloc[-1]:,.0f}")
    print(f"  MCap Trend:         {current_trend_mcap:+.6f} ({'UP' if current_trend_mcap > 0 else 'DOWN'})")
    print(f"  Holder Trend:       {current_trend_holders:+.6f} ({'UP' if current_trend_holders > 0 else 'DOWN'})")
    print(f"  Momentum Active:    {'YES' if in_momentum else 'NO'}")
    print(f"  Entry signals:      {n_entries}")
    print(f"  Exit signals:       {n_exits}")
    print(f"{'='*60}")

    # Determine symbol for plot title
    tokens = load_token_list()
    symbol = address[:8]
    for t in tokens:
        if t["address"] == address:
            symbol = t["symbol"]
            break

    path = plot_analysis(df, symbol, address)
    import subprocess
    subprocess.run(["open", path])


def cmd_scan(args):
    """Scan all tokens for active momentum."""
    tokens = load_token_list()
    active = []

    for token in tokens:
        traj = extract_trajectory(token["address"])
        if traj is None or len(traj) < 10:
            continue

        df = generate_signals(traj, h=args.h, d=args.d)
        if df is None:
            continue

        trend_mcap = df["trend_mcap"].iloc[-1]
        trend_holders = df["trend_holders"].iloc[-1]

        if trend_mcap > 0 and trend_holders > 0:
            active.append({
                "symbol": token["symbol"],
                "address": token["address"],
                "mcap": df["mcap"].iloc[-1],
                "holders": df["holders"].iloc[-1],
                "trend_mcap": trend_mcap,
                "trend_holders": trend_holders,
                "hours": len(traj),
            })

    if not active:
        print("No tokens with active momentum found.")
        return

    active.sort(key=lambda x: x["trend_mcap"], reverse=True)

    print(f"\n{'='*80}")
    print(f"  {len(active)} Tokens with Active Momentum (MCap + Holder trends both positive)")
    print(f"{'='*80}")
    for a in active:
        print(f"  {a['symbol']:12s}  MCap=${a['mcap']:>12,.0f}  Holders={a['holders']:>6,.0f}  "
              f"Trend: mcap={a['trend_mcap']:+.6f}  holders={a['trend_holders']:+.6f}  "
              f"({a['hours']}h)")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Kalman + CUSUM Momentum Detection")
    parser.add_argument("--h", type=float, default=1.0, help="CUSUM threshold (in std units)")
    parser.add_argument("--d", type=float, default=0.0, help="CUSUM drift")
    sub = parser.add_subparsers(dest="command")

    a = sub.add_parser("analyze", help="Analyze a single token")
    a.add_argument("address", help="Contract address")
    a.add_argument("--chain", default="sol")

    sub.add_parser("scan", help="Scan all tokens for active momentum")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "scan":
        cmd_scan(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
