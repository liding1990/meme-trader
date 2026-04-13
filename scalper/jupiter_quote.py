"""Jupiter Quote API — realistic swap price estimation for paper trading.

Uses Jupiter V6 Quote API (free, no key required) to calculate
real swap output amounts and price impact.
"""

import logging

import requests

log = logging.getLogger("scalper.jupiter")

QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"

# Well-known mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

SOL_DECIMALS = 9
USDC_DECIMALS = 6


def get_quote(input_mint: str, output_mint: str, amount: int, slippage_bps: int = 100) -> dict | None:
    """Get a swap quote from Jupiter.

    Args:
        input_mint: input token mint address
        output_mint: output token mint address
        amount: input amount in smallest units (lamports for SOL, etc.)
        slippage_bps: slippage tolerance in basis points (100 = 1%)

    Returns:
        Quote dict with outAmount, priceImpactPct, routePlan, etc.
        None if API fails.
    """
    try:
        resp = requests.get(QUOTE_URL, params={
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": slippage_bps,
        }, timeout=10)

        if resp.status_code != 200:
            log.debug(f"Jupiter quote error: HTTP {resp.status_code}")
            return None

        return resp.json()
    except Exception as e:
        log.debug(f"Jupiter quote failed: {e}")
        return None


def estimate_buy_price(token_mint: str, usd_amount: float = 50.0) -> dict:
    """Estimate buying a token with USDC.

    Returns:
        {
            "price_per_token": float,
            "tokens_received": float,
            "price_impact_pct": float,
            "effective_price": float,  # including slippage
            "success": bool,
        }
    """
    # Convert USD to USDC lamports
    usdc_amount = int(usd_amount * 10**USDC_DECIMALS)

    # USDC → Token
    quote = get_quote(USDC_MINT, token_mint, usdc_amount)
    if not quote:
        return {"success": False, "reason": "no_quote"}

    try:
        out_amount = int(quote.get("outAmount", 0))
        price_impact = float(quote.get("priceImpactPct", "0") or "0")
        in_amount = int(quote.get("inAmount", usdc_amount))

        if out_amount <= 0:
            return {"success": False, "reason": "zero_output"}

        # We don't know token decimals, but we can compute effective price
        effective_price_usd = usd_amount / out_amount if out_amount > 0 else 0

        return {
            "success": True,
            "tokens_received": out_amount,
            "price_impact_pct": price_impact,
            "effective_price": effective_price_usd,
            "raw_quote": quote,
        }
    except Exception as e:
        return {"success": False, "reason": str(e)}


def estimate_sell_price(token_mint: str, token_amount: int) -> dict:
    """Estimate selling tokens for USDC.

    Returns:
        {
            "usdc_received": float,
            "price_impact_pct": float,
            "success": bool,
        }
    """
    quote = get_quote(token_mint, USDC_MINT, token_amount)
    if not quote:
        return {"success": False, "reason": "no_quote"}

    try:
        out_amount = int(quote.get("outAmount", 0))
        price_impact = float(quote.get("priceImpactPct", "0") or "0")

        usdc_received = out_amount / 10**USDC_DECIMALS

        return {
            "success": True,
            "usdc_received": usdc_received,
            "price_impact_pct": price_impact,
            "raw_quote": quote,
        }
    except Exception as e:
        return {"success": False, "reason": str(e)}


def check_liquidity(token_mint: str, trade_size_usd: float = 50.0) -> dict:
    """Check if a token has sufficient liquidity for our trade size.

    Returns:
        {
            "ok": bool,
            "price_impact_pct": float,
            "reason": str,
        }
    """
    result = estimate_buy_price(token_mint, trade_size_usd)

    if not result.get("success"):
        return {"ok": False, "price_impact_pct": 100.0, "reason": result.get("reason", "failed")}

    impact = result["price_impact_pct"]
    ok = abs(impact) < config.MAX_PRICE_IMPACT_PCT

    return {
        "ok": ok,
        "price_impact_pct": impact,
        "reason": "ok" if ok else f"impact={impact:.1f}%>{config.MAX_PRICE_IMPACT_PCT}%",
    }


# Import config here to avoid circular import at module level
from scalper import config
