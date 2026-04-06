"""Shared feature computation for all funnel layers."""

import numpy as np
import pandas as pd


def compute_roc(arr, window):
    """Rate of change over window."""
    r = np.zeros(len(arr))
    for i in range(window, len(arr)):
        if arr[i - window] > 0:
            r[i] = (arr[i] - arr[i - window]) / arr[i - window]
    return r


def compute_rolling_std(arr, window):
    return pd.Series(arr).rolling(window, min_periods=2).std().fillna(0).values


def compute_vwap(mcap, volume, window=24):
    cum_vol = pd.Series(volume).rolling(window, min_periods=1).sum()
    cum_vp = pd.Series(mcap * volume).rolling(window, min_periods=1).sum()
    vwap = cum_vp / cum_vol.replace(0, np.nan)
    return vwap.fillna(pd.Series(mcap)).values


def compute_hmm_states(returns, n_states=3):
    """Fit HMM, return (states, transition_matrix)."""
    from hmmlearn import hmm as _hmm

    clean = returns[~np.isnan(returns)].reshape(-1, 1)
    if len(clean) < 20:
        return np.zeros(len(returns), dtype=int), np.eye(n_states)

    try:
        model = _hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                                  n_iter=50, random_state=42)
        model.fit(clean)
        states = model.predict(clean)

        full = np.zeros(len(returns), dtype=int)
        valid_idx = np.where(~np.isnan(returns))[0]
        full[valid_idx] = states

        means = [clean[states == s].mean() if (states == s).sum() > 0 else 0
                 for s in range(n_states)]
        rank = np.argsort(means)
        remap = {rank[i]: i for i in range(n_states)}
        full = np.array([remap.get(s, s) for s in full])

        return full, model.transmat_
    except Exception:
        return np.zeros(len(returns), dtype=int), np.eye(n_states)


def extract_features_at_tick(mcap, volume, holders, top10, tick_idx,
                              entry_price=None, holding_hours=0,
                              sold_pct=0.0, remaining_pct=1.0, peak_price=None):
    """Extract full feature vector at a single tick.

    If entry_price is None, position-state features are set to 0 (for L2 entry model).
    """
    n = len(mcap)
    i = tick_idx
    feat = {}

    # A. Position state (only if in a trade)
    if entry_price is not None and entry_price > 0:
        feat["unrealized_pnl"] = (mcap[i] - entry_price) / entry_price
        feat["holding_hours"] = holding_hours
        feat["sold_pct"] = sold_pct
        feat["remaining_pct"] = remaining_pct
        pp = peak_price if peak_price else entry_price
        feat["distance_from_peak"] = (pp - mcap[i]) / max(pp, 1)
    else:
        feat["unrealized_pnl"] = 0
        feat["holding_hours"] = 0
        feat["sold_pct"] = 0
        feat["remaining_pct"] = 1.0
        feat["distance_from_peak"] = 0

    # B. Price momentum
    feat["roc_1h"] = compute_roc(mcap, 1)[i] if i >= 1 else 0
    feat["roc_4h"] = compute_roc(mcap, 4)[i] if i >= 4 else 0
    feat["roc_12h"] = compute_roc(mcap, 12)[i] if i >= 12 else 0

    returns = np.diff(mcap[:i+1]) / np.maximum(mcap[:i], 1) if i > 0 else np.array([0])
    feat["volatility_1h"] = np.std(returns[-6:]) if len(returns) >= 6 else np.std(returns) if len(returns) > 1 else 0
    feat["volatility_4h"] = np.std(returns[-4:]) if len(returns) >= 4 else feat["volatility_1h"]

    # Volume
    vol_window = volume[max(0, i-12):i+1]
    feat["volume_trend"] = 0
    if len(vol_window) >= 4:
        mid = len(vol_window) // 2
        v1 = vol_window[:mid].mean()
        v2 = vol_window[mid:].mean()
        feat["volume_trend"] = (v2 - v1) / max(v1, 1)

    feat["volume_concentration"] = 0
    if len(vol_window) > 0 and vol_window.sum() > 0:
        feat["volume_concentration"] = vol_window.max() / vol_window.sum()

    # C. VWAP
    vwap = compute_vwap(mcap[:i+1], volume[:i+1], min(24, i+1))
    feat["price_vs_vwap"] = mcap[i] / max(vwap[-1], 1)
    feat["vwap_slope"] = (vwap[-1] - vwap[-2]) / max(vwap[-2], 1) if len(vwap) >= 2 else 0

    # D. HMM (must be precomputed and passed via hmm_precomputed kwarg)
    # Skip HMM in per-tick extraction — too slow. Use precompute_hmm_for_token() instead.
    feat["hmm_state"] = 0
    feat["hmm_state_duration"] = 0
    feat["hmm_trans_to_down"] = 0.33

    # E. On-chain
    if holders is not None and len(holders) > i:
        feat["holder_growth_1h"] = (holders[i] - holders[max(0,i-1)]) / max(holders[max(0,i-1)], 1)
        feat["holder_growth_4h"] = (holders[i] - holders[max(0,i-4)]) / max(holders[max(0,i-4)], 1) if i >= 4 else 0
        feat["mcap_per_holder"] = mcap[i] / max(holders[i], 1)
    else:
        feat["holder_growth_1h"] = 0
        feat["holder_growth_4h"] = 0
        feat["mcap_per_holder"] = mcap[i]

    if top10 is not None and len(top10) > i:
        feat["top10_pct"] = top10[i]
        feat["top10_change"] = top10[i] - top10[max(0,i-1)]
    else:
        feat["top10_pct"] = np.nan
        feat["top10_change"] = np.nan

    # F. Early warning (placeholder, filled by caller)
    feat["ew_predict_mult"] = 1.0
    feat["ew_grade"] = 5

    return feat


def precompute_hmm_for_token(mcap):
    """Precompute HMM states for entire token at once. Returns (states, durations, trans_to_down)."""
    n = len(mcap)
    returns = np.concatenate([[0], np.diff(mcap) / np.maximum(mcap[:-1], 1)])
    states, transmat = compute_hmm_states(returns)

    durations = np.ones(n)
    for i in range(1, n):
        if states[i] == states[i-1]:
            durations[i] = durations[i-1] + 1

    trans_to_down = np.array([transmat[s][0] if s < len(transmat) else 0.33 for s in states])

    return states, durations, trans_to_down


def apply_hmm_to_features(feat, hmm_states, hmm_durations, hmm_trans, tick_idx):
    """Apply precomputed HMM values to a feature dict."""
    feat["hmm_state"] = int(hmm_states[tick_idx])
    feat["hmm_state_duration"] = hmm_durations[tick_idx]
    feat["hmm_trans_to_down"] = hmm_trans[tick_idx]


FEATURE_NAMES = [
    "unrealized_pnl", "holding_hours", "sold_pct", "remaining_pct", "distance_from_peak",
    "roc_1h", "roc_4h", "roc_12h", "volatility_1h", "volatility_4h",
    "volume_trend", "volume_concentration",
    "price_vs_vwap", "vwap_slope",
    "hmm_state", "hmm_state_duration", "hmm_trans_to_down",
    "holder_growth_1h", "holder_growth_4h", "mcap_per_holder",
    "top10_pct", "top10_change",
    "ew_predict_mult", "ew_grade",
]
