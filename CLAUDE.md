# Meme Trader

## Project Overview

Memecoin Token DNA Fingerprinting & Clustering System.

Given a Memecoin contract address, the system fetches four historical time-series curves that together form the token's "DNA fingerprint":

1. **Price & Volume** — historical OHLCV data
2. **Holders** — number of unique holding addresses over time
3. **Top 10 Concentration** — total supply percentage held by top 10 addresses over time
4. **Market Cap / Holders** — per-holder market cap ratio over time

These four curves are encoded into a vector embedding and projected into a 3D space for visualization and clustering analysis.

### Technical Pipeline

```
Raw 4D Time Series → TS2Vec (encoding) → UMAP (3D projection) → HDBSCAN (clustering)
```

- **TS2Vec**: Contrastive learning encoder that produces fixed-length embeddings from variable-length multivariate time series
- **UMAP**: Non-linear dimensionality reduction preserving local cluster structure
- **HDBSCAN**: Density-based clustering that auto-discovers cluster count and identifies noise/outliers

## Python Environment

**MUST use `venv` to manage the Python environment.**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Always activate the venv before running any Python commands. Do not use conda, poetry, or other environment managers.
