# tools.py
from __future__ import annotations

import math
import time
from typing import Any, Dict

import pandas as pd
import requests


BINANCE_BASE = "https://api.binance.com"


def _http_get(url: str, params: dict | None = None, timeout: int = 20) -> Any:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> Dict[str, Any]:
    """
    Fetch OHLCV candles from Binance (public endpoint, no API key).
    Returns a JSON-friendly dict.
    """
    symbol = symbol.upper().strip()
    endpoint = f"{BINANCE_BASE}/api/v3/klines"
    data = _http_get(endpoint, params={"symbol": symbol, "interval": interval, "limit": int(limit)})

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    # Convert types
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    candles = df[["open_time", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    return {
        "source": "binance",
        "symbol": symbol,
        "interval": interval,
        "limit": int(limit),
        "candles": candles,
        "fetched_at_unix": int(time.time()),
    }


def compute_indicators(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> Dict[str, Any]:
    """
    Fetch candles + compute indicators: SMA(20), SMA(50), RSI(14), volatility (last 50 returns).
    """
    payload = get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(payload["candles"])
    if df.empty or df["close"].isna().all():
        return {"error": "No candle data available."}

    close = df["close"].astype(float)
    ret = close.pct_change()

    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]

    # RSI(14)
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
        "symbol": payload["symbol"],
        "interval": payload["interval"],
        "latest_price": latest_price,
        "sma_20": None if math.isnan(sma_20) else float(sma_20),
        "sma_50": None if math.isnan(sma_50) else float(sma_50),
        "rsi_14": None if pd.isna(rsi_14) else float(rsi_14),
        "volatility_50": None if pd.isna(vol_50) else float(vol_50),
        "trend_hint": trend_hint,
        "n_candles": int(len(df)),
    }


def risk_label(volatility_50: float | None, rsi_14: float | None) -> Dict[str, Any]:
    """
    Deterministic risk label (informational only).
    """
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
