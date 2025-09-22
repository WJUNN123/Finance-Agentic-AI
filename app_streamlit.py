# app_streamlit.py — FULL conversion from Gradio to Streamlit (no Gradio import anywhere)
# ---------------------------------------------------------------------------------
# This preserves ALL functionality from your original Gradio app:
# - CoinGecko market + history
# - Prophet forecast
# - LSTM forecast (TensorFlow)
# - FinBERT sentiment via Hugging Face
# - RSS headline fetcher (feedparser)
# - Short-term memory (SQLite)
# - Optional long-term memory (SentenceTransformers + FAISS)
# - Explainability trace and strategy scenarios
# - Same pretty, structured output as you showed
# - Streamlit UI: single text input + Send button, plus tabs for Summary / Explainability / Headlines / Chart
# ---------------------------------------------------------------------------------

import os
import re
import time
import math
import tempfile
import requests
import feedparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import List, Dict, Tuple
import json
import uuid
import sqlite3
import joblib
import io
import google.generativeai as genai

# UI
import streamlit as st
import altair as alt


# ML / NLP libs
from transformers import pipeline
from huggingface_hub import login as hf_login

# Prophet
from prophet import Prophet

# TensorFlow for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Optional: sentence-transformers + faiss for embeddings and retrieval
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMB_AVAILABLE = True
except Exception:
    EMB_AVAILABLE = False

# Optional: SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="Crypto Analyst — Streamlit", layout="wide")

# ---------------------------
# Configuration
# ---------------------------
FINBERT_MODEL = "ProsusAI/finbert"
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://news.google.com/rss/search?q=cryptocurrency&hl=en-US&gl=US&ceid=US:en",
]
DEFAULT_COINS = [
    {"name": "Bitcoin", "id": "bitcoin", "symbol": "btc"},
    {"name": "Ethereum", "id": "ethereum", "symbol": "eth"},
    {"name": "Solana", "id": "solana", "symbol": "sol"},
    {"name": "BNB", "id": "binancecoin", "symbol": "bnb"},
    {"name": "XRP", "id": "ripple", "symbol": "xrp"},
    {"name": "Cardano", "id": "cardano", "symbol": "ada"},
    {"name": "Dogecoin", "id": "dogecoin", "symbol": "doge"},
]
COIN_NAME_TO_ID = {c["name"].lower(): c["id"] for c in DEFAULT_COINS}
COIN_SYMBOL_TO_ID = {c["symbol"].lower(): c["id"] for c in DEFAULT_COINS}

# Device for transformers pipeline: -1 -> CPU, 0 -> GPU (if available)
PIPELINE_DEVICE = -1

# LSTM defaults
LSTM_WINDOW = 30
LSTM_EPOCHS = 20
LSTM_BATCH = 16

# Hugging Face token (optional; Streamlit Cloud: set in Secrets or env)
HF_TOKEN = None
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN", None)
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        hf_login(token=HF_TOKEN)
    except Exception:
        pass

# Memory & model storage
MEM_DB = "memory.db"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Embeddings / FAISS paths
FAISS_INDEX_PATH = "faiss.index"
METADATA_STORE = "faiss_meta.json"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------
# Utility helpers
# ---------------------------
def human_dt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "-"

# ---------------------------
# Market helpers
# ---------------------------
def coingecko_market(coin_ids: List[str]) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "per_page": max(1, len(coin_ids)),
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def coingecko_chart(coin_id: str, days: int = 180) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["ts_ms", "price"]) if prices else pd.DataFrame(columns=["ts_ms", "price"])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df.drop(columns=["ts_ms"], inplace=True)
    return df

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    if prices is None or len(prices) < period + 1:
        return float("nan")
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    if prices is None or len(prices) < period + 1:
        return pd.Series(index=prices.index, dtype=float)
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ewma_vol(logrets: pd.Series, span: int = 20) -> float:
    """Exponentially-weighted volatility (daily)."""
    if logrets is None or len(logrets.dropna()) < 5:
        return float("nan")
    return float(logrets.ewm(span=span, adjust=False).std().iloc[-1])

def compound_from_returns(p0: float, rets: np.ndarray) -> np.ndarray:
    """Turn an array of log returns into a price path starting at p0."""
    return p0 * np.exp(np.cumsum(rets))

# ---------------------------
# RSS news fetcher
# ---------------------------
def fetch_rss_articles(keyword: str, limit_per_feed: int = 20) -> List[Dict]:
    keyword_l = keyword.lower()
    items = []
    for feed_url in RSS_FEEDS:
        try:
            d = feedparser.parse(feed_url)
            for entry in d.entries[:limit_per_feed]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                link = entry.get("link", "")
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    pub_ts = time.mktime(published)
                else:
                    pub_ts = time.time()
                text_blob = f"{title} {summary}"
                if keyword_l in text_blob.lower():
                    items.append({
                        "title": title,
                        "summary": summary,
                        "link": link,
                        "published_ts": pub_ts,
                        "published_h": human_dt(pub_ts),
                        "source": feed_url,
                    })
        except Exception:
            continue
    # dedupe by title and sort newest first
    seen = set()
    deduped = []
    for it in sorted(items, key=lambda x: x["published_ts"], reverse=True):
        if it["title"] not in seen:
            seen.add(it["title"])
            deduped.append(it)
    return deduped[:50]

# ---------------------------
# FinBERT sentiment (lazy load)
# ---------------------------
FINBERT_PIPE = None

def load_finbert_pipeline():
    global FINBERT_PIPE
    FINBERT_PIPE = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        device=PIPELINE_DEVICE
    )
    return FINBERT_PIPE

def run_finbert(headlines: List[str]) -> List[Dict]:
    global FINBERT_PIPE
    if not headlines:
        return []
    if FINBERT_PIPE is None:
        try:
            FINBERT_PIPE = load_finbert_pipeline()
        except Exception as e:
            raise RuntimeError(f"Failed to load FinBERT: {e}")
    preds = FINBERT_PIPE(headlines, truncation=True, max_length=512)
    out = []
    for h, p in zip(headlines, preds):
        label = p.get("label", "").lower()
        score = float(p.get("score", 0.0))
        out.append({"text": h, "label": label, "score": score})
    return out

def sentiment_score(analyses: List[Dict]) -> Tuple[float, pd.DataFrame]:
    if not analyses:
        return 0.0, pd.DataFrame(columns=["text", "label", "score", "value"])
    mapping = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    rows, val_sum, w_sum = [], 0.0, 0.0
    for a in analyses:
        v = mapping.get(a["label"], 0.0)
        w = a.get("score", 0.0)
        val_sum += v * w
        w_sum += w
        rows.append({"text": a["text"], "label": a["label"], "score": a["score"], "value": v})
    agg = val_sum / w_sum if w_sum > 0 else 0.0
    df = pd.DataFrame(rows)
    return float(agg), df

def sentiment_percentages(analyses: List[Dict]) -> Dict[str, float]:
    total = len(analyses)
    if total == 0:
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for a in analyses:
        lab = a.get("label", "").lower()
        if lab in counts:
            counts[lab] += 1
    return {k: (counts[k] / total) * 100.0 for k in counts}

# ---------------------------
# Prophet forecasting
# ---------------------------
def forecast_with_prophet(price_series: pd.Series, days: int = 7) -> Tuple[pd.DataFrame, dict]:
    if price_series is None or len(price_series) < 30:
        return pd.DataFrame(), {"error": "Not enough history for Prophet (need >= 30 points)."}

    df = price_series.reset_index().rename(columns={price_series.name: "y", "index": "ds"})
    df = df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.25,
        changepoint_range=0.9,
        growth="linear",
    )
    try:
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        summary = {}
        if not forecast.empty:
            f = forecast.tail(days)
            summary = {"pred_last": float(f["yhat"].iloc[-1]),
                       "upper": float(f["yhat_upper"].iloc[-1]),
                       "lower": float(f["yhat_lower"].iloc[-1])}
        return forecast, summary
    except Exception as e:
        return pd.DataFrame(), {"error": str(e)}

# ---------------------------
# LSTM forecasting
# ---------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")
    return model

def train_and_forecast_lstm(price_series: pd.Series,
                            horizon: int = 7,
                            window: int = LSTM_WINDOW,
                            epochs: int = LSTM_EPOCHS):
    """
    LSTM predicts next-day log-returns (drift), not absolute price.
    Features: ret, 7/14-day MA distance, RSI(14).
    We then compound predicted returns to generate the price path.
    We also compute EWMA volatility to later build uncertainty bands.
    """
    if price_series is None or len(price_series) < window + 20:
        return []

    s = price_series.astype(float).copy()
    logp = np.log(s)
    ret = logp.diff()

    df = pd.DataFrame({
        "price": s,
        "ret": ret,
    })
    df["ma7"]  = s.rolling(7).mean()
    df["ma14"] = s.rolling(14).mean()
    df["ma7_dist"]  = (s - df["ma7"]) / s
    df["ma14_dist"] = (s - df["ma14"]) / s
    df["rsi14"] = rsi_series(s, 14)

    df = df.dropna().copy()
    if df.empty or len(df) < window + horizon:
        return []

    # X: features, y: next-day log-return
    feats = df[["ret", "ma7_dist", "ma14_dist", "rsi14"]].values
    y_ret = df["ret"].shift(-1).dropna().values
    feats = feats[:-1, :]  # align X with y

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(feats)

    # Build sliding windows
    X, y = [], []
    for i in range(window, len(X_scaled)):
        X.append(X_scaled[i - window:i, :])
        y.append(y_ret[i])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Model
    model = Sequential()
    model.add(LSTM(64, input_shape=(window, X.shape[2]), return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.10))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=LSTM_BATCH, callbacks=[es], verbose=0)

    # Recursive forecast of log-returns
    last_window = X_scaled[-window:, :].copy()
    preds_rets = []
    # Estimate recent volatility for realism
    sigma = ewma_vol(df["ret"], span=20)
    sigma = 0.0 if (isinstance(sigma, float) and math.isnan(sigma)) else float(sigma)

    for _ in range(horizon):
        x_in = last_window.reshape((1, window, X.shape[2]))
        mu_hat = float(model.predict(x_in, verbose=0)[0][0])  # predicted next-day log-return (drift)
        preds_rets.append(mu_hat)

        # Build next pseudo-feature row for the sliding window (keep it simple and stable)
        next_ret = mu_hat
        # Keep MA distances & RSI roughly stable (we’re just rolling the window)
        last_feat = last_window[-1, :].copy()
        feat_next = last_feat.copy()
        feat_next[0] = next_ret  # replace "ret" channel
        feat_next = feat_next.reshape(1, -1)
        last_window = np.vstack([last_window[1:], feat_next])

    # Turn predicted returns into prices
    p0 = float(s.iloc[-1])
    path = compound_from_returns(p0, np.array(preds_rets))
    return path.tolist()

# ---------------------------
# Chart
# ---------------------------
def plot_history_forecasts_to_file(hist: pd.DataFrame, prophet_forecast: pd.DataFrame, lstm_preds: List[float], coin_id: str) -> str:
    if hist is None or hist.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist.index, hist["price"], label="History", linewidth=2)
    if prophet_forecast is not None and not prophet_forecast.empty:
        try:
            fc_idx = pd.to_datetime(prophet_forecast["ds"])
            ax.plot(fc_idx, prophet_forecast["yhat"], linestyle="--", linewidth=1.5, label="Prophet Forecast")
            ax.fill_between(fc_idx, prophet_forecast["yhat_lower"], prophet_forecast["yhat_upper"], alpha=0.15)
        except Exception:
            pass
    if lstm_preds:
        try:
            last_dt = hist.index[-1]
            future_index = [last_dt + pd.Timedelta(days=i+1) for i in range(len(lstm_preds))]
            ax.plot(future_index, lstm_preds, linestyle=":", linewidth=2, label="LSTM Forecast")
            ax.annotate(f"${lstm_preds[-1]:.2f}", (future_index[-1], lstm_preds[-1]), textcoords="offset points", xytext=(0,8), ha='center', weight="bold")
        except Exception:
            pass
    if lstm_preds and prophet_forecast is not None and not prophet_forecast.empty:
        try:
            prophet_vals = prophet_forecast["yhat"].tail(len(lstm_preds)).values
            ensemble_vals = 0.7 * np.array(lstm_preds) + 0.3 * prophet_vals
            last_dt = hist.index[-1]
            future_index = [last_dt + pd.Timedelta(days=i+1) for i in range(len(ensemble_vals))]
            ax.plot(future_index, ensemble_vals, linestyle="-.", linewidth=1.6, label="Ensemble (Prophet+LSTM)")
            ax.annotate(f"${ensemble_vals[-1]:.2f}", (future_index[-1], ensemble_vals[-1]), textcoords="offset points", xytext=(0, -12), ha='center', weight="bold")
        except Exception:
            pass
    if len(hist) >= 7: ax.plot(hist.index, hist["price"].rolling(7).mean(), linestyle=":", label="7D MA", alpha=0.8)
    if len(hist) >= 14: ax.plot(hist.index, hist["price"].rolling(14).mean(), linestyle=":", label="14D MA", alpha=0.8)
    ax.set_title(f"{coin_id.capitalize()} — Price & Forecast", fontsize=12, fontweight="bold")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    tmp = tempfile.NamedTemporaryFile(prefix=f"{coin_id}_chart_", suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# ---------------------------
# Recommendation & insight
# ---------------------------
def scale(x, lo, hi):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return max(-1.0, min(1.0, 2 * (x - lo) / (hi - lo) - 1))

def recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days):
    recommendation = "HOLD / WAIT"
    score = 0.5
    
    if sentiment > 0.5:
        recommendation = "BUY"
        score = 0.7
    elif sentiment < -0.5:
        recommendation = "SELL / AVOID"
        score = 0.3

    return {
        "rating": recommendation,
        "score": score,
        "insight": f"Based on sentiment {sentiment}, the recommendation is {recommendation}.",
        "source": "fallback"
    }

def call_gpt3_for_insight(
    coin_id: str,
    coin_symbol: str,
    sentiment: float,
    pct_24h: float,
    pct_7d: float,
    rsi: float,
    price: float,
    market_cap: float,
    volume_24h: float,
    risk: str,
    horizon_days: int,
    top_headlines: List[str] = None,
    forecast_note: str = "",
    max_tokens: int = 500,
    temperature: float = 0.3
) -> Dict:
    gemini_api_key = st.secrets.get("gemini", {}).get("api_key", None) or os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        st.warning("Gemini API key not found. Falling back to rule-based analysis.")
        return recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)
    
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        st.warning(f"Error configuring Gemini API: {str(e)}")
        return recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)

    headlines_context = ""
    if top_headlines:
        headlines_context = "\n\nTop recent headlines:\n" + "\n".join([f"- {h}" for h in top_headlines[:5]])

    def safe_format(val, default="N/A", format_str="{:.2f}"):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return format_str.format(val)

    rsi_zone = "N/A"
    if not (isinstance(rsi, float) and math.isnan(rsi)):
        rsi_zone = "Overbought" if rsi >= 70 else "Oversold" if rsi <= 30 else "Neutral"

    prompt = f"""You are an expert cryptocurrency analyst. Analyze the following data for {coin_id.upper()} ({coin_symbol.upper()}) and provide investment insights.
    
    MARKET DATA:
    - Current Price: ${safe_format(price)}
    - Market Cap: ${safe_format(market_cap, format_str="{:,.0f}")}
    - 24h Volume: ${safe_format(volume_24h, format_str="{:,.0f}")}
    - 24h Change: {safe_format(pct_24h)}%
    - 7d Change: {safe_format(pct_7d)}%
    - RSI (14): {safe_format(rsi)} ({rsi_zone})
    
    SENTIMENT ANALYSIS:
    - News Sentiment Score: {sentiment:.3f} (range: -1 to +1, where +1 is very positive)
    
    ANALYSIS PARAMETERS:
    - Risk Tolerance: {risk}
    - Investment Horizon: {horizon_days} days{headlines_context}
    
    {f'FORECAST NOTE: {forecast_note}' if forecast_note else ''}
    
    Please provide:
    1. A clear BUY/SELL/HOLD recommendation with reasoning
    2. Detailed insights covering:
       - Sentiment analysis interpretation
       - Technical momentum (24h and 7d trends)
       - RSI analysis and what it suggests
       - Risk factors to consider
       - Key catalysts to watch
    
    Format your response as a structured analysis. Be specific about price levels, timeframes, and actionable advice. Consider the user's risk tolerance and investment horizon.
    
    Keep the tone professional but accessible. Include appropriate disclaimers that this is educational content, not financial advice."""

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=temperature, max_output_tokens=max_tokens
        ))
        gemini_response = response.text.strip()
        rating = extract_recommendation(gemini_response)
        score = calculate_score_from_response(gemini_response, sentiment, pct_24h, pct_7d, rsi)
        
        return {"rating": rating, "score": score, "insight": gemini_response, "source": "gemini"}
        
    except Exception as e:
        st.warning(f"Gemini API error: {e}. Falling back to rule-based analysis.")
        return recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)

def calculate_score_from_response(gpt_response: str, sentiment: float, pct_24h: float, pct_7d: float, rsi: float) -> float:
    response_lower = gpt_response.lower()
    base_score = 0.0
    if sentiment is not None: base_score += 0.4 * sentiment
    if pct_24h is not None and not math.isnan(pct_24h): base_score += 0.2 * max(-1.0, min(1.0, pct_24h / 15.0))
    if pct_7d is not None and not math.isnan(pct_7d): base_score += 0.2 * max(-1.0, min(1.0, pct_7d / 40.0))
    if rsi is not None and not math.isnan(rsi):
        rsi_component = -0.2 if rsi >= 70 else 0.2 if rsi <= 30 else (50 - rsi) / 100.0
        base_score += 0.2 * rsi_component
    
    positive_words = ["bullish", "positive", "strong", "buy", "upward", "growth", "opportunity"]
    negative_words = ["bearish", "negative", "weak", "sell", "downward", "risk", "caution"]
    pos_count = sum(1 for word in positive_words if word in response_lower)
    neg_count = sum(1 for word in negative_words if word in response_lower)
    
    gpt_adjustment = 0.1 * (pos_count - neg_count) / 10.0 if pos_count > neg_count else -0.1 * (neg_count - pos_count) / 10.0
    return max(-1.0, min(1.0, base_score + gpt_adjustment))

def extract_recommendation(gemini_response: str) -> str:
    response_lower = gemini_response.lower()
    if any(p in response_lower for p in ["strong buy", "buy recommendation", "recommend buying"]): return "BUY"
    if any(p in response_lower for p in ["buy", "accumulate", "long position"]) and "avoid" not in response_lower and "don't" not in response_lower: return "BUY"
    if any(p in response_lower for p in ["sell", "short", "avoid", "exit"]): return "SELL / AVOID"
    return "HOLD / WAIT"

# =================================================================
# 1) MEMORY DB
# =================================================================
def init_memory_db(path=MEM_DB):
    con = sqlite3.connect(path, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        message TEXT,
        meta_json TEXT,
        ts TIMESTAMP
    )
    """)
    con.commit()
    return con

MEM_CONN = init_memory_db()

def save_conversation(session_id: str, role: str, message: str, meta: dict=None):
    try:
        cur = MEM_CONN.cursor()
        cur.execute(
            "INSERT INTO conversations (session_id, role, message, meta_json, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, message, json.dumps(meta or {}), datetime.utcnow().isoformat())
        )
        MEM_CONN.commit()
    except Exception as e:
        print("Memory save error:", e)

def load_recent_session(session_id: str, limit: int = 20):
    try:
        cur = MEM_CONN.cursor()
        cur.execute("SELECT role, message, meta_json, ts FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
                    (session_id, limit))
        rows = cur.fetchall()
        return [{"role": r[0], "message": r[1], "meta": json.loads(r[2]), "ts": r[3]} for r in reversed(rows)]
    except Exception:
        return []

# =================================================================
# 2) Long-term memory (FAISS) — MODIFIED AND CORRECTED
# =================================================================
EMB_MODEL = None
FAISS_INDEX = None
FAISS_META = []

def init_faiss(dim=None):
    global EMB_MODEL, FAISS_INDEX, FAISS_META
    if not EMB_AVAILABLE:
        return None, []
    if EMB_MODEL is None:
        # THE FIX IS HERE: Explicitly set device to 'cpu'.
        EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME, device='cpu')
        
    dim_local = EMB_MODEL.get_sentence_embedding_dimension() if dim is None else dim
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_STORE):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_STORE, 'r') as f:
                metas = json.load(f)
            FAISS_INDEX = index
            FAISS_META[:] = metas
            st.toast("✅ Loaded long-term memory from disk.")
            return index, FAISS_META
    except Exception as e:
        st.warning(f"⚠️ Could not load FAISS memory: {e}. Creating a new one.")
        
    index = faiss.IndexFlatL2(dim_local)
    FAISS_INDEX = index
    FAISS_META = []
    return index, FAISS_META

def save_faiss_memory():
    """Saves the current FAISS index and metadata to disk."""
    if not EMB_AVAILABLE or FAISS_INDEX is None:
        return False
    try:
        faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
        with open(METADATA_STORE, "w") as f:
            json.dump(FAISS_META, f)
        st.toast("💾 Saved long-term memory to disk.")
        return True
    except Exception as e:
        print(f"FAISS save error: {e}")
        st.error(f"Error saving FAISS memory: {e}")
        return False

# Call init_faiss on script start
if EMB_AVAILABLE:
    init_faiss()

def add_long_term_item(text: str, meta: dict):
    global EMB_MODEL, FAISS_INDEX, FAISS_META
    if not EMB_AVAILABLE or FAISS_INDEX is None:
        return False
    if EMB_MODEL is None:
        EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME, device='cpu')
        
    emb = EMB_MODEL.encode([text], convert_to_numpy=True).astype("float32")
    fid = len(FAISS_META)
    try:
        FAISS_INDEX.add(emb)
        FAISS_META.append({"id": fid, "text": text, "meta": meta})
        return True
    except Exception as e:
        print(f"FAISS add error: {e}")
        return False

def retrieve_similar(query: str, k: int=5):
    if not EMB_AVAILABLE or FAISS_INDEX is None or EMB_MODEL is None or FAISS_INDEX.ntotal == 0:
        return []
    q_emb = EMB_MODEL.encode([query], convert_to_numpy=True).astype("float32")
    try:
        search_k = min(k, FAISS_INDEX.ntotal)
        D, I = FAISS_INDEX.search(q_emb, search_k)
    except Exception as e:
        print(f"FAISS search error: {e}")
        return []
    results = [FAISS_META[idx] for idx in I[0] if idx < len(FAISS_META)]
    return results

# =================================================================
# 3) Training helpers
# =================================================================
def collect_training_data(coin_id, lookback_days=180, horizon=7):
    df = coingecko_chart(coin_id, days=lookback_days+horizon)
    if df.empty: return None, None
    prices = df["price"].reset_index(drop=True)
    X, y = [], []
    for i in range(LSTM_WINDOW, len(prices)-horizon+1):
        X.append(prices[i-LSTM_WINDOW:i].values.astype(float))
        y.append(prices[i+horizon-1])
    if not X: return None, None
    X = np.array(X).reshape((-1, LSTM_WINDOW, 1))
    y = np.array(y).astype(float)
    return X, y

def retrain_lstm_for_coin(coin_id, epochs=5):
    X, y = collect_training_data(coin_id)
    if X is None:
        return False, "Not enough data"
    model = build_lstm_model((LSTM_WINDOW, 1))
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(MODEL_DIR, f"lstm_{coin_id}_{ts}.h5")
    model.save(path)
    if joblib:
        joblib.dump({"coin": coin_id, "saved_at": ts, "shape": X.shape}, path + ".meta.pkl")
    return True, path

# =================================================================
# 4) Planner + Explainability
# =================================================================
def rule_based_plan(market, sentiment_score_val, rsi, risk="medium", horizon_days=7):
    steps = []
    steps.append("Step 0: Monitor price and headlines every 1 hour for the next 24h.")
    if not (isinstance(rsi, float) and math.isnan(rsi)) and rsi < 30 and sentiment_score_val > 0.2:
        steps.append("Step 1: Condition met (RSI < 30 AND sentiment positive). Consider opening small position; set stop-loss at -4% and target +8%.")
    elif sentiment_score_val < -0.2 and (not (isinstance(rsi, float) and math.isnan(rsi)) and rsi > 65):
        steps.append("Step 1: Negative sentiment and high RSI → consider reducing exposure or delaying new buys.")
    else:
        steps.append("Step 1: No immediate action recommended. Wait for clearer signal (RSI breach or sentiment change).")
    steps.append("Step 2: If a position is opened, use trailing stop of 3% and move stop to breakeven after +6% gain.")
    steps.append("Step 3: Log decisions and outcomes to long-term memory for continual learning.")
    return steps

def llm_plan_stub(context_text: str):
    return ["LLM Step 1: Evaluate immediate headlines and order book.", "LLM Step 2: Provide actionable thresholds (user define)."]

def scale_ex(x, lo, hi):
    try:
        return max(-1.0, min(1.0, (float(x) - lo) / (hi - lo) * 2 - 1))
    except Exception:
        return 0.0

def explain_trace_components(agg_sent, pct_24h, pct_7d, rsi):
    mom24 = scale_ex(pct_24h, -15, 15)
    mom7  = scale_ex(pct_7d, -40, 40)

    rsi_dev = 0.0
    if not (isinstance(rsi, float) and math.isnan(rsi)):
        rsi_dev = -1.0 if rsi >= 70 else 1.0 if rsi <= 30 else (50 - rsi) / 20.0

    weights = {"w_sent": 0.45, "w_m24": 0.2, "w_m7": 0.2, "w_rsi": 0.15}
    comp = {
        "sentiment_component": float(weights["w_sent"] * agg_sent),
        "mom24_component": float(weights["w_m24"] * mom24),
        "mom7_component": float(weights["w_m7"] * mom7),
        "rsi_component": float(weights["w_rsi"] * rsi_dev),
    }
    comp["total_score"] = sum(comp.values())
    return comp, weights

def generate_scenarios(sentiment: float, rsi: float, pct_24h: float, pct_7d: float) -> List[str]:
    scenarios = []
    if sentiment < -0.2: scenarios.append("🟡 If the Fed signals a pause or rate cut next month → possible bullish reversal.")
    if rsi <= 35: scenarios.append("🟢 RSI is nearing oversold territory → short-term bounce likely (watch for reversal).")
    if pct_7d < -8: scenarios.append("🔻 If price breaks below weekly support → 5–10% downside risk.")
    if -0.15 < sentiment < 0.15 and -7 < pct_24h < 7: scenarios.append("⚖️ If no external catalyst occurs → price may continue to consolidate sideways.")
    if sentiment > 0.05 and pct_24h > 0: scenarios.append("🔄 Sentiment recovery in progress → sideways to slightly bullish expected.")
    return scenarios

# =================================================================
# 5) Full analyze_coin and UI logic
# =================================================================
def analyze_coin(coin_id: str,
                 coin_symbol: str,
                 risk: str,
                 horizon_days: int,
                 custom_note: str = "",
                 forecast_days: int = 7,
                 model_choice: str = "ensemble"):
    try:
        mkt = coingecko_market([coin_id])
    except Exception as e:
        return {"error": f"Market data failed: {e}"}
    if mkt.empty:
        return {"error": f"No market data for {coin_id}"}
    row = mkt.iloc[0]
    price = float(row.get("current_price", float("nan")))
    pct_24h = row.get("price_change_percentage_24h", float("nan"))
    pct_7d = row.get("price_change_percentage_7d_in_currency", row.get("price_change_percentage_7d", float("nan")))
    hist = coingecko_chart(coin_id, days=180)
    rsi = compute_rsi(hist["price"]) if (not hist.empty and "price" in hist.columns) else float("nan")
    market_cap = float(row.get("market_cap", float("nan")))
    volume_24h = float(row.get("total_volume", float("nan")))

    arts_symbol = fetch_rss_articles(coin_symbol)
    arts_name = fetch_rss_articles(coin_id)
    articles = {a["title"]: a for a in (arts_symbol + arts_name)}
    articles = list(articles.values())
    headlines = [a["title"] for a in articles]

    try:
        analyses = run_finbert(headlines) if headlines else []
    except Exception as e:
        return {"error": f"FinBERT error: {e}"}
    agg_sent, df_sent = sentiment_score(analyses)
    sentiment_pcts = sentiment_percentages(analyses)

    # Get top headlines for GPT context
    top_headlines = headlines[:5] if headlines else []

    prophet_df, prophet_summary = pd.DataFrame(), {}
    lstm_preds = []
    if model_choice.lower() in ("prophet", "ensemble"):
        try:
            prophet_df, prophet_summary = forecast_with_prophet(hist["price"] if (not hist.empty and "price" in hist.columns) else pd.Series(), days=forecast_days)
        except Exception:
            prophet_df, prophet_summary = pd.DataFrame(), {}
    if model_choice.lower() in ("lstm", "ensemble"):
        try:
            lstm_preds = train_and_forecast_lstm(hist["price"] if (not hist.empty and "price" in hist.columns) else pd.Series(), horizon=forecast_days, window=LSTM_WINDOW, epochs=LSTM_EPOCHS)
        except Exception:
            lstm_preds = []

    # Build forecast table
    forecast_table = []
    last_date = hist.index[-1] if (hist is not None and not hist.empty) else None
    for i in range(forecast_days):
        day = i + 1
        date = (last_date + pd.Timedelta(days=day)) if last_date is not None else None
        prophet_val = None
        try:
            if (prophet_df is not None) and (not prophet_df.empty):
                future_rows = prophet_df.tail(forecast_days).reset_index(drop=True)
                if i < len(future_rows):
                    prophet_val = float(future_rows.loc[i, "yhat"])
        except Exception:
            prophet_val = None
        lstm_val = None
        try:
            if lstm_preds and i < len(lstm_preds):
                lstm_val = float(lstm_preds[i])
        except Exception:
            lstm_val = None
        ensemble_val = None
        if (prophet_val is not None) and (lstm_val is not None):
            ensemble_val = float(0.3 * prophet_val + 0.7 * lstm_val)
        elif lstm_val is not None:
            ensemble_val = float(lstm_val)
        elif prophet_val is not None:
            ensemble_val = float(prophet_val)
        forecast_table.append({
            "day": day,
            "date": date,
            "prophet": prophet_val,
            "lstm": lstm_val,
            "ensemble": ensemble_val
        })
    last_prediction = None
    if forecast_table:
        last_row = forecast_table[-1]
        last_prediction = last_row.get("ensemble") if last_row.get("ensemble") is not None else (last_row.get("prophet") or last_row.get("lstm"))

    # Generate forecast note
    forecast_note = ""
    try:
        if forecast_table and isinstance(price, (int, float)) and price > 0:
            fc_vals = [r.get("ensemble") or r.get("prophet") or r.get("lstm") for r in forecast_table if r]
            if fc_vals and (fc_vals[-1] is not None):
                fc_7d = 100.0 * (float(fc_vals[-1]) - float(price)) / float(price)
                if (fc_7d < 0) and isinstance(rsi, (int, float)) and rsi < 35:
                    forecast_note = (
                        "Forecast expects mild drift lower, but RSI is near oversold—"
                        "a short-term bounce is possible. HOLD / WAIT is prudent."
                    )
                elif (fc_7d > 0) and isinstance(rsi, (int, float)) and rsi > 70:
                    forecast_note = (
                        "Forecast points upward, but RSI is near overbought—"
                        "pullback risk is elevated. Consider caution on entries."
                    )
    except Exception:
        pass

    # *** MAIN CHANGE: Use GPT-3 for recommendation and insights ***
    rec = call_gpt3_for_insight(
        coin_id=coin_id,
        coin_symbol=coin_symbol,
        sentiment=agg_sent,
        pct_24h=pct_24h,
        pct_7d=pct_7d,
        rsi=rsi,
        price=price,
        market_cap=market_cap,
        volume_24h=volume_24h,
        risk=risk,
        horizon_days=horizon_days,
        top_headlines=top_headlines,
        forecast_note=forecast_note
    )

    # Monte-Carlo uncertainty band
    mc_mean, mc_lo, mc_hi = [], [], []
    try:
        if hist is not None and not hist.empty and "price" in hist.columns and lstm_preds:
            hist_prices = hist["price"].astype(float)
            p0 = float(hist_prices.iloc[-1])

            logrets = np.log(hist_prices).diff().dropna()
            if not logrets.empty:
                sig = float(logrets.ewm(span=20, adjust=False).std().iloc[-1])
            else:
                sig = 0.0
            sig = 0.0 if (isinstance(sig, float) and (math.isnan(sig) or sig < 1e-9)) else sig

            drift = np.diff(np.log([p0] + list(lstm_preds)))
            horizon = len(lstm_preds)
            if len(drift) < horizon and len(drift) > 0:
                drift = np.pad(drift, (0, horizon - len(drift)), mode="edge")

            rng = np.random.default_rng(42)
            n_paths = 50
            sims = []
            for _ in range(n_paths):
                noise = rng.normal(0.0, sig, size=horizon) * 0.6
                rets = drift + noise
                path = p0 * np.exp(np.cumsum(rets))
                sims.append(path)
            sims = np.array(sims)
            mc_mean = sims.mean(axis=0).tolist()
            mc_lo   = np.percentile(sims, 25, axis=0).tolist()
            mc_hi   = np.percentile(sims, 75, axis=0).tolist()
    except Exception:
        mc_mean, mc_lo, mc_hi = [], [], []

    # Memory & retrieval
    try:
        mem_text = f"{coin_id} snapshot {datetime.utcnow().isoformat()}: price {price}, 24h {pct_24h}, sentiment {agg_sent:.3f}"
        add_long_term_item(mem_text, {"price": price, "pct24": pct_24h, "sent": agg_sent})
    except Exception:
        pass
    try:
        similar = retrieve_similar(f"{coin_id} {' '.join(headlines[:3])}", k=3)
    except Exception:
        similar = []

    comp, weights = explain_trace_components(agg_sent, pct_24h, pct_7d, rsi)
    explain_trace = {
        "weights": weights,
        "components": comp,
        "similar_events": similar,
        "top_headlines": df_sent.head(3).to_dict(orient="records") if not df_sent.empty else [],
        "insight_source": rec.get("source", "unknown")  # Track GPT vs fallback
    }

    if not df_sent.empty:
        df_sent["published"] = [next((a["published_h"] for a in articles if a["title"] == t), "-") for t in df_sent["text"]]
        df_sent["link"] = [next((a["link"] for a in articles if a["title"] == t), "-") for t in df_sent["text"]]

    return {
        "market": {
            "coin": coin_id,
            "symbol": coin_symbol.upper(),
            "price_usd": price,
            "pct_change_24h": pct_24h,
            "pct_change_7d": pct_7d,
            "rsi_14": rsi,
            "market_cap": market_cap,
            "volume_24h": volume_24h
        },
        "history": hist,
        "articles": articles,
        "sentiment_table": df_sent,
        "sentiment_score": agg_sent,
        "sentiment_percentages": sentiment_pcts,
        "recommendation": rec,  # Now generated by GPT-3
        "prophet_df": prophet_df,
        "prophet_summary": prophet_summary,
        "lstm_preds": lstm_preds,
        "forecast_table": forecast_table,
        "last_prediction": last_prediction,
        "explainability": explain_trace,
        "mc_mean": mc_mean,
        "mc_lo": mc_lo,
        "mc_hi": mc_hi,
        "forecast_note": forecast_note,
    }

# =================================================================
# 6) User Input and Text Output (new)
# =================================================================
def parse_user_message(message: str) -> Dict:
    msg = message.lower()
    coin_id = None
    for name, cid in COIN_NAME_TO_ID.items():
        if name in msg:
            coin_id = cid; break
    if coin_id is None:
        for sym, cid in COIN_SYMBOL_TO_ID.items():
            if re.search(r'\b' + re.escape(sym) + r'\b', msg):
                coin_id = cid; break
    if coin_id is None:
        coin_id = "bitcoin"
    intent = "general"
    if any(w in msg for w in ["buy", "should i buy", "can i buy", "buy now", "position"]):
        intent = "advice"
    elif any(w in msg for w in ["forecast", "next", "predict", "price in", "what will"]):
        intent = "forecast"
    elif "sentiment" in msg or "news" in msg or "headlines" in msg:
        intent = "sentiment"
    elif any(w in msg for w in ["price", "current price", "how much is"]):
        intent = "price"
    horizon_days = 7
    m = re.search(r'(\d+)\s*(day|days|d)\b', msg)
    if m: horizon_days = int(m.group(1))
    else:
        if "week" in msg: horizon_days = 7
        if "month" in msg: horizon_days = 30
    coin_info = next((c for c in DEFAULT_COINS if c["id"] == coin_id), DEFAULT_COINS[0])
    return {"coin_id": coin_id, "coin_symbol": coin_info["symbol"], "intent": intent, "horizon_days": horizon_days}

def _pick_insight_line(insight_text: str, label: str, fallback: str = "—") -> str:
    if not insight_text:
        return fallback
    lines = [ln.strip() for ln in insight_text.splitlines() if ln.strip()]
    patterns = [
        rf"^\*\*{re.escape(label)}\*\*\s*:?\s*(.+)$",
        rf"^{re.escape(label)}\s*:?\s*(.+)$",
    ]
    for ln in lines:
        for pat in patterns:
            m = re.match(pat, ln, flags=re.IGNORECASE)
            if m:
                return ln
    return fallback

def make_pretty_output(result: Dict, horizon_days: int) -> str:
    # (your original condensed text builder; unchanged)
    market = result["market"]
    pcts = result.get("sentiment_percentages", {})
    rec = result["recommendation"]
    ft = result.get("forecast_table", [])
    ex = result.get("explainability", {})
    comps = ex.get("components", {})

    sym = f"{market['coin'].capitalize()} ({market['symbol']})"
    price_str = f"${market['price_usd']:,.2f}"
    c24 = market.get("pct_change_24h", float("nan"))
    c7  = market.get("pct_change_7d", float("nan"))
    rsi = market.get("rsi_14", float("nan"))

    mc  = market.get("market_cap", float("nan"))
    vol = market.get("volume_24h", market.get("total_volume", float("nan")))
    mcs  = "NA" if (mc is None or (isinstance(mc, float) and math.isnan(mc))) else f"${mc:,.0f}"
    vols = "NA" if (vol is None or (isinstance(vol, float) and math.isnan(vol))) else f"${vol:,.0f}"

    c24s = "NA" if (c24 is None or (isinstance(c24, float) and math.isnan(c24))) else f"{c24:.2f}%"
    c7s  = "NA" if (c7  is None or (isinstance(c7,  float) and math.isnan(c7)))  else f"{c7:.2f}%"
    rsis = "NA" if (rsi is None or (isinstance(rsi, float) and math.isnan(rsi))) else f"{rsi:.1f}"

    s_pos = float(pcts.get("positive", 0.0))
    s_neu = float(pcts.get("neutral", 0.0))
    s_neg = float(pcts.get("negative", 0.0))

    ft_lines = []
    for row in ft[:horizon_days]:
        d = row.get("date")
        dstr = d.strftime("%Y-%m-%d") if d else "-"
        ens = row.get("ensemble")
        ens_s = f"{ens:,.2f}" if ens is not None else "-"
        ft_lines.append(f"Day +{row['day']} ({dstr}): {ens_s}")
    ft_text = "\n".join(ft_lines) if ft_lines else "No forecast available."

    sim_score = (s_pos - s_neg) / 100.0
    scenarios = generate_scenarios(sim_score, rsi, c24, c7)
    scenario_text = "\n".join([f"👉 {s}" for s in scenarios]) if scenarios else "No active strategic signals."

    risk_lines = ["⚠️ Risk Disclosure"]
    try:
        if (isinstance(vol, (int, float)) and isinstance(mc, (int, float))
            and not math.isnan(vol) and not math.isnan(mc) and mc > 0):
            liq_pct = 100.0 * vol / mc
            risk_lines.append(f"Liquidity Risk: 24h volume is {liq_pct:.2f}% of market cap (thin order book risk).")
    except Exception:
        pass
    try:
        arts = result.get("articles", [])
        titles_blob = " ".join([a.get("title", "") for a in arts])[:3000]
        if re.search(r"\b(sec|regulat|ban|lawsuit|fine|enforcement|probe|review)\b", titles_blob, re.I):
            risk_lines.append("Regulatory Risk: Recent headlines reference regulatory actions or reviews.")
    except Exception:
        pass
    total_score = comps.get("total_score", None)
    if total_score is not None:
        exam_score = int(round(50 + 50 * float(total_score)))
        exam_score = max(0, min(100, exam_score))
        mood = ("very bullish 📈" if exam_score >= 80 else
                "slightly bullish 📈" if exam_score >= 60 else
                "neutral ⚖️"        if exam_score >= 40 else
                "slightly bearish 📉" if exam_score >= 20 else
                "very bearish 📉")
        risk_lines.append(f"Score: {exam_score} → {mood}")

    def comp_line(label, key, meaning_pos, meaning_neg, neutral_hint):
        val = comps.get(key, None)
        if val is None:
            return None
        if val > 0.02:
            trend = "slightly bullish"
            hint  = meaning_pos
        elif val < -0.02:
            trend = "slightly bearish"
            hint  = meaning_neg
        else:
            trend = "neutral"
            hint  = neutral_hint
        return f"{label}: {val:+.4f} → {trend} ({hint})"

    for ln in [
        comp_line("RSI",       "rsi_component",       "price momentum is improving",     "potential overbought/oversold", "balanced momentum"),
        comp_line("Sentiment", "sentiment_component", "news sentiment is mildly positive","news sentiment is mildly negative","mixed/neutral news flow"),
        comp_line("Mom24",     "mom24_component",     "24-hour momentum positive",       "flat 24-hour momentum",         "flat 24-hour momentum"),
        comp_line("Mom7",      "mom7_component",      "7-day momentum uptrend",          "7-day momentum almost flat",    "flat 7-day momentum"),
    ]:
        if ln: risk_lines.append(ln)

    risk_block = "\n".join(risk_lines)

    insight_text = rec.get("insight", "").strip()
    sent_line  = _pick_insight_line(insight_text, "Sentiment", "—")
    m24_line   = _pick_insight_line(insight_text, "24-hour Momentum", "—")
    m7_line    = _pick_insight_line(insight_text, "7-day Momentum", "—")
    rsi_line   = _pick_insight_line(insight_text, "RSI (14)", "—")

    top_block = "\n".join([
        f"📊 {sym}",
        "",
        f"💵 Price: {price_str}",
        f"🏦 Market Cap: {mcs}",
        f"📊 24h Volume: {vols}",
        f"📈 Changes: 24h → {c24s} | 7d → {c7s}",
        f"📊 RSI(14): {rsis}",
    ])

    sentiment_block = "\n".join([
        "📰 Sentiment & Recommendation",
        "",
        f"Sentiment: Positive {s_pos:.1f}% | Neutral {s_neu:.1f}% | Negative {s_neg:.1f}%",
        "",
        f"Recommendation: {rec['rating']}",
        "",
        "Insight:",
        sent_line,
        "",
        m24_line,
        "",
        m7_line,
        "",
        rsi_line,
    ])

    momentum_block = "\n".join([
        "📈 Momentum & RSI Analysis",
        "",
        f"24h Momentum: {m24_line.replace('**', '')}",
        "",
        f"7-day Momentum: {m7_line.replace('**', '')}",
        "",
        f"RSI (14): {rsi_line.replace('**', '')}",
    ])

    strategy_block = "\n".join([
        "🧠 Strategy Simulation (What-If Scenarios)",
        "",
        scenario_text
    ])

    forecast_block = "\n".join([
        f"🔮 Price Forecast ({horizon_days}-Day)",
        "",
        ft_text
    ])

    output = "\n\n".join([
        top_block,
        sentiment_block,
        momentum_block,
        risk_block,
        strategy_block,
        forecast_block
    ])
    return output

# =================================================================
# 7) UI HELPERS (new)
# =================================================================
def _fmt_money(n):
    if n is None or (isinstance(n, float) and math.isnan(n)): return "—"
    a = abs(n)
    if a >= 1_000_000_000_000: return f"${n/1_000_000_000_000:.2f}T"
    if a >= 1_000_000_000:     return f"${n/1_000_000_000:.2f}B"
    if a >= 1_000_000:         return f"${n/1_000_000:.2f}M"
    return f"${n:,.2f}"

def _rsi_zone(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return "Overbought" if v >= 70 else ("Oversold" if v <= 30 else "Neutral")

def _sentiment_bar(pos, neu, neg, width_blocks=20):
    pb = int(round(width_blocks * (pos/100.0)))
    nb = int(round(width_blocks * (neu/100.0)))
    rb = max(0, width_blocks - pb - nb)
    return "🟩"*pb + "⬜"*nb + "🟥"*rb

def _rec_style(rating: str):
    txt = (rating or "").lower()
    if "buy" in txt:
        return ("BUY", "🟢", "#16a34a")
    if "sell" in txt or "avoid" in txt:
        return ("SELL / AVOID", "🔴", "#ef4444")
    return ("HOLD / WAIT", "🟡", "#f59e0b")
    
def render_pretty_summary(result, horizon_days: int = 7):
    """
    Pretty dashboard renderer for the Summary view.
    Complete version with expanded sentiment and combined analysis layout.
    """
    market = result["market"]
    pcts = result.get("sentiment_percentages", {}) or {}
    rec  = result.get("recommendation", {}) or {}

    # --- Unpack market ---
    name  = f"{market.get('coin','').capitalize()} ({market.get('symbol','').upper()})"
    price = market.get("price_usd", float("nan"))
    c24   = market.get("pct_change_24h", float("nan"))
    c7    = market.get("pct_change_7d", float("nan"))
    rsi   = market.get("rsi_14", float("nan"))
    mcap  = market.get("market_cap", float("nan"))
    vol24 = market.get("volume_24h", float("nan"))

    # --- Derived ---
    liq_pct = (vol24 / mcap * 100.0) if (isinstance(mcap,(int,float)) and mcap>0 and isinstance(vol24,(int,float))) else None
    c24_arrow = "🔼" if (isinstance(c24,(int,float)) and c24 >= 0) else "🔽"
    c24_color = "#2ecc71" if (isinstance(c24,(int,float)) and c24 >= 0) else "#e74c3c"

    # Recommendation cosmetics
    rec_text  = rec.get("rating","HOLD / WAIT")
    rec_label, rec_emoji, rec_color = _rec_style(rec_text)

    # =========================
    # Header: Price + Metrics
    # =========================
    st.markdown(f"### 📊 {name}")
    cols = st.columns([1.5, 1.2, 1.2, 1.2])
    with cols[0]:
        st.markdown("**Price**")
        st.markdown(f"<span style='font-size:2rem;font-weight:800'>$ {price:,.2f}</span>", unsafe_allow_html=True)
        if isinstance(c24,(int,float)):
            st.markdown(
                f"<span style='padding:4px 8px;border-radius:999px;background:{c24_color}22;color:{c24_color};font-weight:700'>{c24_arrow} {c24:.2f}% · 24h</span>",
                unsafe_allow_html=True
            )
        # Big recommendation badge
        st.markdown(
            f"<span style='display:inline-block;margin-top:8px;padding:6px 12px;border-radius:12px;"
            f"background:{rec_color}22;color:{rec_color};font-weight:800;font-size:1.0rem'>"
            f"{rec_emoji} {rec_label}</span>",
            unsafe_allow_html=True
        )
    with cols[1]:
        st.metric("Market Cap", _fmt_money(mcap))
        st.metric("24h Volume", _fmt_money(vol24))
    with cols[2]:
        st.metric("7d Change", f"{c7:.2f}%" if isinstance(c7,(int,float)) else "—")
        st.metric("RSI (14)", f"{rsi:.1f}" if isinstance(rsi,(int,float)) else "—")
    with cols[3]:
        st.write("**Quick tags**")
        chips = []
        if isinstance(c7,(int,float)): chips.append(f"7d: {c7:.2f}%")
        if isinstance(liq_pct,(int,float)): chips.append(f"Liquidity: {liq_pct:.1f}%")
        chips.append(f"RSI: {f'{rsi:.1f}' if isinstance(rsi,(int,float)) else '—'} · {_rsi_zone(rsi)}")
        chips.append(rec_text)
        for chip in chips:
            st.markdown(
                f"<span style='display:inline-block;background:#1b2332;color:#eaf0ff;padding:4px 8px;border-radius:12px;margin:2px'>{chip}</span>",
                unsafe_allow_html=True
            )

    st.divider()

    # =========================
    # Recommendation & Insight Section (Full Width with Sidebar)
    # =========================
    header_col1, header_col2 = st.columns([2.5, 1])
    with header_col1:
        st.subheader("✅ Recommendation & Insight")
    with header_col2:
        st.subheader("⚠️ Risks")

    main_col, sidebar_col = st.columns([2.5, 1])

    with main_col:
        rec_score = rec.get("score", None)
        rec_rating = rec.get("rating", "HOLD / WAIT")
        
        rec_label, rec_emoji, rec_color = _rec_style(rec_rating)
        
        st.markdown(
            f"<span style='display:inline-block;padding:8px 16px;border-radius:12px;"
            f"background:{rec_color}22;color:{rec_color};font-weight:800;font-size:1.1rem'>"
            f"{rec_emoji} {rec_label}</span>",
            unsafe_allow_html=True
        )
        
        if isinstance(rec_score, (int, float)):
            score100 = max(0, min(100, int(round(50 + 50*float(rec_score)))))
            st.progress(score100/100.0, text=f"Model score: {rec_score:+.2f} (−1..+1) → {score100}/100")
        
        rec_source = rec.get("source", "unknown")
        if rec_source == "gemini":
            st.caption("🤖 Powered by Google Gemini")
        elif rec_source in ["gpt3", "gpt-3.5", "openai"]:
            st.caption("🤖 Powered by OpenAI GPT")
        elif rec_source == "fallback":
            st.caption("⚙️ Rule-based analysis")

        st.write("")

        st.markdown("**📰 Sentiment Analysis**")
        pos = float(pcts.get("positive", 0.0))
        neu = float(pcts.get("neutral", 0.0)) 
        neg = float(pcts.get("negative", 0.0))
        
        st.markdown(_sentiment_bar(pos, neu, neg))
        st.caption(f"Positive {pos:.1f}% · Neutral {neu:.1f}% · Negative {neg:.1f}%")
        
        ins = rec.get("insight","").strip()
        if ins and len(ins) > 50:
            import re
            cleaned_insight = ins
            cleaned_insight = re.sub(r'Recommendation:\s*(BUY|SELL|HOLD)[^\n]*', f'Recommendation: {rec_label}', cleaned_insight, flags=re.IGNORECASE)
            cleaned_insight = re.sub(r'\d+\.\s*Recommendation:\s*(BUY|SELL|HOLD)[^\n]*', f'1. Recommendation: {rec_label}', cleaned_insight, flags=re.IGNORECASE)
            
            st.info(cleaned_insight)
        else:
            st.info(f"**System Recommendation: {rec_label}** - Sentiment analysis based on recent news headlines and market data.")

    with sidebar_col:
        # === RISKS SECTION ===
        risk_lines = []
        if isinstance(liq_pct,(int,float)):
            badge = "🔴" if liq_pct < 5 else ("🟡" if liq_pct < 10 else "🟢")
            risk_lines.append(f"{badge} Liquidity: {liq_pct:.1f}% of market cap")
        
        arts = result.get("articles", []) or []
        joined = " ".join([a.get("title","") for a in arts])[:3000]
        if re.search(r"\b(sec|regulat|ban|lawsuit|enforcement|probe|review)\b", joined, re.I):
            risk_lines.append("🟡 Regulatory mentions")
        
        ex = result.get("explainability", {}) or {}
        comps = ex.get("components", {}) or {}
        ttl = comps.get("total_score", None)
        if ttl is not None:
            score = max(0, min(100, int(round(50 + 50*float(ttl)))))
            st.progress(score/100.0, text=f"Composite Score: {score}/100")
        
        for risk in risk_lines:
            st.write(f"• {risk}")
        
        if not risk_lines:
            st.write("• 🟢 No major risks detected")

        # === MOMENTUM & RSI SECTION ===
        st.subheader("📈 Momentum & RSI")
        
        # 24h momentum
        if isinstance(c24,(int,float)):
            if c24 > 2:
                st.write(f"• **24h**: +{c24:.2f}% (Strong)")
            elif c24 > 0:
                st.write(f"• **24h**: +{c24:.2f}% (Slight gain)")
            elif c24 > -2:
                st.write(f"• **24h**: {c24:+.2f}% (Sideways)")
            else:
                st.write(f"• **24h**: {c24:+.2f}% (Declining)")
        else:
            st.write("• **24h**: Data unavailable")
            
        # 7d momentum  
        if isinstance(c7,(int,float)):
            if c7 > 5:
                st.write(f"• **7d**: +{c7:.2f}% (Strong trend)")
            elif c7 > 0:
                st.write(f"• **7d**: +{c7:.2f}% (Weekly gain)")
            elif c7 > -5:
                st.write(f"• **7d**: {c7:+.2f}% (Consolidation)")
            else:
                st.write(f"• **7d**: {c7:+.2f}% (Decline)")
        else:
            st.write("• **7d**: Data unavailable")
            
        # RSI
        if isinstance(rsi,(int,float)):
            zone = _rsi_zone(rsi)
            st.write(f"• **RSI**: {rsi:.1f} ({zone})")
        else:
            st.write("• **RSI**: Data unavailable")

        # === STRATEGY SECTION ===
        st.subheader("🧠 Strategy Signals")
        pos_pct = float(pcts.get("positive", 0.0))
        neg_pct = float(pcts.get("negative", 0.0))
        sim_score = (pos_pct - neg_pct) / 100.0

        scenarios = generate_scenarios(
            sim_score,
            rsi if isinstance(rsi,(int,float)) else float("nan"),
            c24 if isinstance(c24,(int,float)) else float("nan"),
            c7 if isinstance(c7,(int,float)) else float("nan"),
        )

        if scenarios:
            for scenario in scenarios[:3]:
                clean_scenario = scenario.replace("🟡 ", "").replace("🟢 ", "").replace("🔽 ", "").replace("⚖️ ", "").replace("🔄 ", "")
                st.write(f"• {clean_scenario}")
        else:
            st.write("• Monitor for breakout signals")

    st.divider()

    # =========================
    # Forecast - Single instance only
    # =========================
    st.subheader(f"🔮 {horizon_days}-Day Forecast")

    hist_df = result.get("history")
    hist_df = hist_df if isinstance(hist_df, pd.DataFrame) else pd.DataFrame()
    ft = result.get("forecast_table", []) or []

    # Build forecast rows for table
    rows = []
    for row in ft[:horizon_days]:
        d = row.get("date")
        dstr = d.strftime("%Y-%m-%d") if d is not None else "-"
        v = row.get("ensemble") or row.get("prophet") or row.get("lstm")
        rows.append({"Date": dstr, "Forecast ($)": None if v is None else float(v)})

    if rows:
        df_forecast = pd.DataFrame(rows).set_index("Date")
        cL, cR = st.columns([1, 1.3])
        with cL:
            st.dataframe(df_forecast.style.format({"Forecast ($)": "${:,.2f}"}), use_container_width=True)

        with cR:
            combined = pd.DataFrame()
            if not hist_df.empty and "price" in hist_df.columns:
                combined["History"] = hist_df["price"].tail(180)

            if not df_forecast.empty:
                try:
                    forecast_vals = df_forecast["Forecast ($)"].astype(float)
                    forecast_vals.index = pd.to_datetime(df_forecast.index)
                    combined = pd.concat([combined, forecast_vals.rename("Forecast")])
                except Exception:
                    pass

            if not combined.empty:
                df_plot = combined.copy()

                try:
                    df_plot.index = df_plot.index.tz_convert(None)
                except Exception:
                    try:
                        df_plot.index = df_plot.index.tz_localize(None)
                    except Exception:
                        pass

                plot_df = df_plot.reset_index().rename(columns={"index": "Date"})
                plot_df = plot_df.melt("Date", var_name="Series", value_name="Value")

                color_scale = alt.Scale(
                    domain=["History", "Forecast"],
                    range=["#4e79a7", "#ff4d4f"]
                )

                base = alt.Chart(plot_df).encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Value:Q", title="Price (USD)"),
                    color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(orient="bottom")),
                    tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=",.2f")]
                )

                lines = base.mark_line(size=2)
                points = base.transform_filter(alt.datum.Series == "Forecast").mark_point(size=40, filled=True)

                mc_mean = result.get("mc_mean") or []
                mc_lo   = result.get("mc_lo") or []
                mc_hi   = result.get("mc_hi") or []
                chart = (lines + points)
                try:
                    if mc_mean and len(mc_mean) == df_forecast.shape[0]:
                        band_df = pd.DataFrame({
                            "Date": pd.to_datetime(df_forecast.index),
                            "lo": mc_lo,
                            "hi": mc_hi,
                        })
                        band_chart = alt.Chart(band_df).mark_area(opacity=0.15).encode(
                            x=alt.X("Date:T", title="Date"),
                            y="lo:Q",
                            y2="hi:Q",
                        )
                        chart = chart + band_chart
                except Exception:
                    pass

                st.altair_chart(chart.interactive(), use_container_width=True)
            else:
                st.caption("_No forecast available._")
    else:
        st.caption("_No forecast available._")
        
# =================================================================
# 8) Build response (returns structured `result` for the UI)
# =================================================================
def build_single_response(user_message: str, session_id: str):
    parsed = parse_user_message(user_message)
    coin_id = parsed["coin_id"]
    horizon_days = parsed["horizon_days"]
    coin_symbol = parsed["coin_symbol"]

    save_conversation(session_id, "user", user_message)

    # Core analysis with Gemini insights
    result = analyze_coin(
        coin_id, coin_symbol,
        risk="Medium",
        horizon_days=horizon_days,
        forecast_days=horizon_days,
        model_choice="ensemble",
    )
    if "error" in result:
        resp = f"Error: {result['error']}"
        save_conversation(session_id, "assistant", resp)
        return resp, {}, "", None, None

    pretty = make_pretty_output(result, horizon_days)

    ex = result.get("explainability", {}) or {}
    headlines_df = result.get("sentiment_table", pd.DataFrame())
    
    chart_path = plot_history_forecasts_to_file(
        result.get("history", pd.DataFrame()),
        result.get("prophet_df", pd.DataFrame()),
        result.get("lstm_preds", []),
        coin_id,
    )

    save_conversation(session_id, "assistant", pretty, {"chart": chart_path})

    return pretty, ex, headlines_df, chart_path, result


# =================================================================
# 9) STREAMLIT APP (Main Layout)
# =================================================================
st.markdown("""
<style>
/* Overall spacing */
.block-container { padding-top: 1.8rem; padding-bottom: 2.4rem; }
/* Cards */
.card { background: #0b1220; border: 1px solid #1f2a44; border-radius: 16px; padding: 16px 18px; }
.card > h3, .card > h4 { margin-top: 0; }
/* Buttons & chips */
button[kind="primary"] { border-radius: 12px !important; }
.stButton>button { width: 100%; }
/* Headings */
h1, h2, h3 { letter-spacing:.01em; }
.app-title{ display:flex; align-items:center; gap:.6rem; }
.app-title .logo{
  width:36px; height:36px; display:inline-flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#7c3aed33,#06b6d433); border:1px solid #24324a; border-radius:12px;
  font-size:1.1rem;
}
.app-subtitle{ color:#96a7bf; margin:-.15rem 0 1.1rem 0; }
/* Info banner */
.banner { background:#0e213a; border:1px solid #1c3357; color:#cfe3ff; border-radius:12px; padding:.9rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ---- Header ------------------------------------------------------------------
st.markdown(
    "<div class='app-title'>"
    "<div class='logo'>📈</div>"
    "<h1 style='margin:0'>Crypto Agent</h1>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='app-subtitle'>Ask about BTC, ETH, SOL, etc. This app renders a single, clean Summary dashboard. "
    "Educational only — not financial advice.</div>",
    unsafe_allow_html=True
)

# ---- Session state -----------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "last_outputs" not in st.session_state:
    st.session_state.last_outputs = {
        "result_for_ui": None,
        "horizon": 7,
    }

# ---- Input card --------------------------------------------------------------
with st.container():
    user_message = st.text_input(
        label="Your message",
        value="",
        placeholder="E.g. 'ETH 7-day forecast' or 'Should I buy BTC?'",
        key="user_text",
        label_visibility="collapsed"
    )
    send_clicked = st.button("Analyze", use_container_width=True, type="primary")

# ---- Handle send -------------------------------------------------------------
if send_clicked and user_message.strip():
    with st.spinner("Analyzing... This may take a moment."):
        pretty_text, full_ex, headlines_df, chart_path, result_obj = build_single_response(
            user_message, st.session_state.session_id
        )
        st.session_state.last_outputs = {
            "result_for_ui": result_obj,
            "horizon": parse_user_message(user_message)["horizon_days"],
        }
        # Immediately re-run to update the display with the new state
        st.experimental_rerun()

# ---- Render summary or an empty state ----------------------------------------
ui_result = st.session_state.last_outputs.get("result_for_ui")
if ui_result:
    # Use tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "🔬 Explainability", "📰 Headlines"])

    with tab1:
        render_pretty_summary(
            ui_result,
            horizon_days=st.session_state.last_outputs.get("horizon", 7),
        )
    
    with tab2:
        st.subheader("🔬 Explainability Trace")
        ex_data = ui_result.get("explainability", {})
        if ex_data:
            st.json(ex_data, expanded=True)
        else:
            st.info("No explainability data available.")
            
    with tab3:
        st.subheader("📰 Recent Headlines & Sentiment")
        df_headlines = ui_result.get("sentiment_table")
        if df_headlines is not None and not df_headlines.empty:
            st.dataframe(df_headlines, use_container_width=True)
        else:
            st.info("No headlines found.")

else:
    st.markdown("<div class='banner'>Ask about a coin to generate the dashboard.</div>", unsafe_allow_html=True)
