# tools.py
from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def _http_get(url: str, params: dict | None = None, timeout: int = 20) -> Any:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "streamlit-agentic-app"})
    r.raise_for_status()
    return r.json()


def _binance_klines(symbol: str, interval: str, limit: int) -> Dict[str, Any]:
    endpoint = f"{BINANCE_BASE}/api/v3/klines"
    data = _http_get(endpoint, params={"symbol": symbol, "interval": interval, "limit": int(limit)})

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
        "symbol": symbol,
        "interval": interval,
        "limit": int(limit),
        "candles": candles,
        "fetched_at_unix": int(time.time()),
    }


def _cg_map_interval_to_days(interval: str, limit: int) -> int:
    # Rough mapping to get enough points. CoinGecko returns per-minute only for short ranges.
    interval = interval.lower()
    if interval in ["1m", "3m", "5m", "15m", "30m"]:
        return 1  # minute granularity available for ~1 day window (varies)
    if interval in ["1h", "2h", "4h", "6h", "12h"]:
        return max(3, min(30, int(limit / 24) + 3))
    if interval in ["1d"]:
        return max(30, min(365, limit + 10))
    return 7


def _cg_coin_id_from_symbol(symbol: str) -> str | None:
    s = symbol.upper()
    # minimal mapping for your demo
    if s.startswith("BTC"):
        return "bitcoin"
    if s.startswith("ETH"):
        return "ethereum"
    return None


def _coingecko_ohlc(symbol: str, interval: str, limit: int) -> Dict[str, Any]:
    coin_id = _cg_coin_id_from_symbol(symbol)
    if not coin_id:
        return {"error": f"CoinGecko fallback only supports BTC/ETH in this demo. Got symbol={symbol}."}

    days = _cg_map_interval_to_days(interval, limit)

    # CoinGecko OHLC endpoint (returns [timestamp, open, high, low, close])
    endpoint = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    data = _http_get(endpoint, params={"vs_currency": "usd", "days": days})

    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["volume"] = None  # OHLC endpoint doesn't provide volume

    # Downsample to approximate requested interval
    # Convert to time index and resample if interval >= 1h
    df = df.set_index("open_time").sort_index()

    rule_map = {
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H", "1d": "1D",
        "15m": "15T", "30m": "30T",
    }
    rule = rule_map.get(interval.lower())
    if rule:
        ohlc = df[["open", "high", "low", "close"]].resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
    else:
        ohlc = df[["open", "high", "low", "close"]].dropna()

    ohlc = ohlc.tail(limit).reset_index()
    candles = ohlc.assign(volume=None).to_dict(orient="records")

    return {
        "source": "coingecko",
        "symbol": symbol,
        "interval": interval,
        "limit": int(limit),
        "candles": candles,
        "fetched_at_unix": int(time.time()),
        "note": "CoinGecko provides USD candles; volume may be missing.",
    }


def get_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> Dict[str, Any]:
    """
    Primary: Binance. Fallback: CoinGecko (BTC/ETH demo).
    Returns JSON-friendly candles or an error with details.
    """
    symbol = symbol.upper().strip()
    interval = interval.strip()
    limit = int(limit)

    try:
        return _binance_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        # fallback
        fallback = _coingecko_ohlc(symbol=symbol, interval=interval, limit=limit)
        return {
            "warning": f"Binance failed ({type(e).__name__}: {e}). Using fallback if available.",
            **fallback,
        }


def compute_indicators(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> Dict[str, Any]:
    payload = get_klines(symbol=symbol, interval=interval, limit=limit)
    if "candles" not in payload or not payload["candles"]:
        return {"error": "No candle data available.", "details": payload}

    df = pd.DataFrame(payload["candles"])
    if df.empty or df["close"].isna().all():
        return {"error": "No usable close prices.", "details": payload}

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
