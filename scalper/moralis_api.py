"""Moralis API client — Historical Token Holders at 5-minute granularity."""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("MORALIS_API_KEY", "")
_BASE_URL = "https://solana-gateway.moralis.io"
_RATE_LIMIT_DELAY = 0.25  # 4 req/s safe margin


def get_historical_holders(
    address: str,
    from_date: str,
    to_date: str,
    time_frame: str = "5min",
    network: str = "mainnet",
    limit: int = 100,
    max_pages: int = 20,
) -> list[dict]:
    """Fetch historical holder data for a Solana token.

    Args:
        address: Token mint address
        from_date: Start date (ISO format or unix seconds)
        to_date: End date (ISO format or unix seconds)
        time_frame: One of 1min, 5min, 10min, 30min, 1h, 4h, 12h, 1d, 1w, 1m
        network: mainnet or devnet
        limit: Results per page (max depends on plan)

    Returns:
        List of {timestamp, totalHolders, netHolderChange, holderPercentChange,
                 newHoldersByAcquisition, holdersIn, holdersOut}
    """
    if not _API_KEY:
        raise RuntimeError("MORALIS_API_KEY not set in .env")

    url = f"{_BASE_URL}/token/{network}/holders/{address}/historical"
    headers = {"X-Api-Key": _API_KEY, "Accept": "application/json"}
    params = {
        "timeFrame": time_frame,
        "fromDate": from_date,
        "toDate": to_date,
        "limit": limit,
    }

    all_results = []
    cursor = None
    page = 0

    while page < max_pages:
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("result", [])
        all_results.extend(results)

        cursor = data.get("cursor")
        if not cursor or not results:
            break

        page += 1
        time.sleep(_RATE_LIMIT_DELAY)

    return all_results
