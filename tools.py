# tools.py
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests

BINANCE_BASES = [
    "https://api.binance.com",
    "https://data-api.binance.vision",  # sometimes works better from cloud
]
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

UA = {"User-Agent": "streamlit-agentic-demo/1.0"}


def _http_get(url: str, params: dict | None = None, timeout: int = 20, retries: int = 3) -> Any:
    last_err: Optional[str] = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            # Explicitly handle rate limit / server errors
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)}"
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(last_err or "Unknown HTTP error")


def _binance_klines(symbol: str, interval: str, limit: int) -> Dict[str, Any]:
    symbol = symbol.upper().strip()
    endpoint_path = "/api/v3/klines"
    last_error = None

    for base in BINANCE_BASES:
        try:
            url = f"{base}{endpoint_path}"
            data = _http_get(url, params={"symbol": symbol, "interval": interval, "limit": int(limit)}, timeout=20, retries=3)

            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "num_trades",
                "taker_base_vol", "taker_quote_vol", "ignore"
            ]
            df = pd.DataFrame(data, columns=cols)

            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            candles = df[["open_time", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
            return {
                "source": "binance",
                "binance_base": base,
                "symbol": symbol,
                "interval": interval,
                "limit": int(limit),
                "candles": candles,
                "fetched_at_unix": int(time.time()),
            }

        except Exception as e:
            last_error = f"{base} -> {type(e).__name__}: {str(e)}"

    raise RuntimeError(last_error or "Binance failed")


def _cg_coin_id(symbol: str) -> Optional[str]:
    s = symbol.upper()
    if s.startswith("BTC"):
        return "bitcoin"
    if s.startswith("ETH"):
        return "ethereum"
    return None


def _coingecko_market_chart(symbol: str, interval: str, limit: int) -> Dict[str, Any]:
    """
    More reliable than /ohlc for demos.
    Uses /market_chart to get prices, then resamples.
    """
    coin_id = _cg_coin_id(symbol)
    if not coin_id:
        return {"error": f"CoinGecko fallback supports BTC/ETH only. Got {symbol}."}

    # choose days to cover enough points
    interval = interval.lower()
    if interval in ["15m", "30m", "1h"]:
        days = 2
    elif interval in ["4h", "6h", "12h"]:
        days = 7
    else:
        days = 30

    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    data = _http_get(url, params={"vs_currency": "usd", "days": days}, timeout=20, retries=3)

    prices = data.get("prices", [])
    if not prices:
        return {"error": "CoinGecko returned no prices.", "raw_keys": list(data.keys())}

    df = pd.DataFrame(prices, columns=["open_time", "price"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time").sort_index()

    rule_map = {"15m": "15T", "30m": "30T", "1h": "1H", "4h": "4H", "1d": "1D"}
    rule = rule_map.get(interval, "1H")

    # Build pseudo-OHLC from price series
    ohlc = df["price"].resample(rule).ohlc().dropna().tail(limit)
    ohlc = ohlc.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}).reset_index()
    ohlc["volume"] = None

    candles = ohlc.rename(columns={"open_time": "open_time"}).to_dict(orient="records")
    return {
        "source": "coingecko",
        "symbol": symbol,
        "interval": interval,
        "limit": int(limit),
        "candles": candles,
        "note": "CoinGecko uses USD price series; OHLC is resampled; volume may be missing.",
        "fetched_at_unix": int(time.time()),
    }


def get_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> Dict[str, Any]:
    symbol = symbol.upper().strip()
    interval = interval.strip()
    limit = int(limit)

    try:
        return _binance_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        fallback = _coingecko_market_chart(symbol=symbol, interval=interval, limit=limit)
        return {
            "warning": f"Binance failed: {type(e).__name__}: {str(e)}",
            **fallback,
        }


def compute_indicators(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> Dict[str, Any]:
    try:
        payload = get_klines(symbol=symbol, interval=interval, limit=limit)
        candles = payload.get("candles", [])
        if not candles:
            return {"error": "No candle data returned.", "details": payload}

        df = pd.DataFrame(candles)
        if df.empty or df["close"].isna().all():
            return {"error": "No usable close prices.", "details": payload}

        close = df["close"].astype(float)
        ret = close.pct_change()

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]

        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-12)
        rsi_14 = (100 - (100 / (1 + rs))).iloc[-1]

        vol_50 = ret.tail(50).std()
        latest_price = float(close.iloc[-1])

        trend_hint = "up" if (not math.isnan(sma_20) and not math.isnan(sma_50) and sma_20 > sma_50) else "down_or_flat"

        return {
            "data_source": payload.get("source"),
            "symbol": payload.get("symbol", symbol),
            "interval": interval,
            "latest_price": latest_price,
            "sma_20": None if math.isnan(sma_20) else float(sma_20),
            "sma_50": None if math.isnan(sma_50) else float(sma_50),
            "rsi_14": None if pd.isna(rsi_14) else float(rsi_14),
            "volatility_50": None if pd.isna(vol_50) else float(vol_50),
            "trend_hint": trend_hint,
            "n_candles": int(len(df)),
            "warning": payload.get("warning"),
            "note": payload.get("note"),
        }

    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}


def risk_label(volatility_50: float | None, rsi_14: float | None) -> Dict[str, Any]:
    if volatility_50 is None or rsi_14 is None:
        return {"risk": "unknown", "reason": "Missing indicator values."}

    if volatility_50 > 0.06:
        vol = "high"
    elif volatility_50 > 0.03:
        vol = "medium"
    else:
        vol = "low"

    if rsi_14 >= 70:
        rsi = "overbought"
    elif rsi_14 <= 30:
        rsi = "oversold"
    else:
        rsi = "neutral"

    return {"risk": vol, "momentum": rsi}
