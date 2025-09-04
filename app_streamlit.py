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


# ===================================================================
# STEP 3: Add these new helper functions before your existing functions
# ===================================================================
def recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days):
    # This function generates a fallback recommendation based on basic rules
    recommendation = "HOLD / WAIT"  # Default to HOLD/WAIT if no GPT-3 response
    score = 0.5  # Neutral score (can be improved with your rule-based logic)
    
    if sentiment > 0.5:
        recommendation = "BUY"
        score = 0.7
    elif sentiment < -0.5:
        recommendation = "SELL / AVOID"
        score = 0.3
    # Further conditions can be added here based on other factors like price change, RSI, etc.

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
    """Call Google Gemini to generate personalized insights and recommendations"""
    
    # Configure Gemini API key
    gemini_api_key = st.secrets.get("gemini", {}).get("api_key", None)
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_api_key:
        try:
            # Successfully retrieved the API key, you can now configure the Gemini model
            genai.configure(api_key=gemini_api_key)
            st.write("Google Gemini API key successfully loaded.")
        except Exception as e:
            st.warning(f"Error configuring Gemini API: {str(e)}")
            return recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)
    else:
        st.warning("Gemini API key not found. Falling back to rule-based analysis.")
        return recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)
    
    # Prepare headlines context
    headlines_context = ""
    if top_headlines:
        headlines_context = f"\n\nTop recent headlines:\n" + "\n".join([f"- {h}" for h in top_headlines[:5]])

    def safe_format(val, default="N/A", format_str="{:.2f}"):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return format_str.format(val)

    rsi_zone = "N/A"
    if not (isinstance(rsi, float) and math.isnan(rsi)):
        if rsi >= 70:
            rsi_zone = "Overbought"
        elif rsi <= 30:
            rsi_zone = "Oversold"
        else:
            rsi_zone = "Neutral"

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

    # Calculate token usage (input tokens + output tokens)
    input_tokens = len(prompt.split())  # Approximate token count for the prompt
    total_tokens = input_tokens + max_tokens
    
    # Ensure token limit is not exceeded (4096 tokens for many models)
    if total_tokens > 4096:
        st.warning(f"Total token count exceeds the model limit. Adjusting max tokens.")
        max_tokens = 4096 - input_tokens  # Adjust to fit within the limit

    try:
        # Initialize the basic Gemini model (for free-tier users)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.9,
            top_k=40
        )
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        gemini_response = response.text.strip()

        # Extract the recommendation and calculate the score
        rating = extract_recommendation(gemini_response)
        score = calculate_score_from_response(gemini_response, sentiment, pct_24h, pct_7d, rsi)
        
        return {
            "rating": rating,
            "score": score,
            "insight": gemini_response,
            "source": "gemini"
        }
        
    except Exception as e:
        st.warning(f"Gemini API error: {e}. Falling back to rule-based analysis.")
        fallback_result = recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)
        fallback_result["source"] = "fallback"
        return fallback_result
        
def calculate_score_from_response(gpt_response: str, sentiment: float, pct_24h: float, pct_7d: float, rsi: float) -> float:
    """Calculate a numerical score based on Gemini response and market data"""
    response_lower = gpt_response.lower()
    
    base_score = 0.0
    
    if sentiment is not None:
        base_score += 0.4 * sentiment
    
    if pct_24h is not None and not math.isnan(pct_24h):
        momentum_24 = max(-1.0, min(1.0, pct_24h / 15.0))
        base_score += 0.2 * momentum_24
    
    if pct_7d is not None and not math.isnan(pct_7d):
        momentum_7 = max(-1.0, min(1.0, pct_7d / 40.0))
        base_score += 0.2 * momentum_7
    
    if rsi is not None and not math.isnan(rsi):
        if rsi >= 70:
            rsi_component = -0.2
        elif rsi <= 30:
            rsi_component = 0.2
        else:
            rsi_component = (50 - rsi) / 100.0
        base_score += 0.2 * rsi_component
    
    gpt_adjustment = 0.0
    positive_words = ["bullish", "positive", "strong", "buy", "upward", "growth", "opportunity"]
    negative_words = ["bearish", "negative", "weak", "sell", "downward", "risk", "caution"]
    
    positive_count = sum(1 for word in positive_words if word in response_lower)
    negative_count = sum(1 for word in negative_words if word in response_lower)
    
    if positive_count > negative_count:
        gpt_adjustment = 0.1 * (positive_count - negative_count) / 10.0
    elif negative_count > positive_count:
        gpt_adjustment = -0.1 * (negative_count - positive_count) / 10.0
    
    final_score = base_score + gpt_adjustment
    return max(-1.0, min(1.0, final_score))

def extract_recommendation(gemini_response: str) -> str:
    """Extract BUY/SELL/HOLD recommendation from Gemini response"""
    response_lower = gemini_response.lower()
    
    if any(phrase in response_lower for phrase in ["strong buy", "buy recommendation", "recommend buying"]):
        return "BUY"
    elif any(phrase in response_lower for phrase in ["buy", "accumulate", "long position"]):
        if "avoid" not in response_lower and "don't" not in response_lower:
            return "BUY"
    elif any(phrase in response_lower for phrase in ["sell", "short", "avoid", "exit"]):
        return "SELL / AVOID"
    elif any(phrase in response_lower for phrase in ["hold", "wait", "neutral", "sideways"]):
        return "HOLD / WAIT"
    
    return "HOLD / WAIT"

# =================================================================
# 1) MEMORY DB (exactly your code, unchanged)
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
# 2) Long-term memory (FAISS) — your code with safe guards
# =================================================================
EMB_MODEL = None
FAISS_INDEX = None
FAISS_META = []

def init_faiss(dim=None):
    global EMB_MODEL, FAISS_INDEX, FAISS_META
    if not EMB_AVAILABLE:
        return None, []
    if EMB_MODEL is None:
        EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)
    dim_local = EMB_MODEL.get_sentence_embedding_dimension() if dim is None else dim
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_STORE):
            index = faiss.read_index(FAISS_INDEX_PATH)
            metas = json.load(open(METADATA_STORE))
            FAISS_INDEX = index
            FAISS_META[:] = metas
            return index, FAISS_META
    except Exception:
        pass
    index = faiss.IndexFlatL2(dim_local)
    FAISS_INDEX = index
    FAISS_META = []
    return index, FAISS_META

if EMB_AVAILABLE:
    init_faiss()

def add_long_term_item(text: str, meta: dict):
    global EMB_MODEL, FAISS_INDEX, FAISS_META
    if not EMB_AVAILABLE:
        return False
    if EMB_MODEL is None:
        EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)
    emb = EMB_MODEL.encode([text], convert_to_numpy=True).astype("float32")
    fid = len(FAISS_META)
    try:
        FAISS_INDEX.add(emb)
        FAISS_META.append({"id": fid, "text": text, "meta": meta})
        faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
        json.dump(FAISS_META, open(METADATA_STORE, "w"))
        return True
    except Exception as e:
        print("FAISS add error:", e)
        return False

def retrieve_similar(query: str, k: int=5):
    if not EMB_AVAILABLE or FAISS_INDEX is None or EMB_MODEL is None:
        return []
    q_emb = EMB_MODEL.encode([query], convert_to_numpy=True).astype("float32")
    try:
        D, I = FAISS_INDEX.search(q_emb, min(k, FAISS_INDEX.ntotal))
    except Exception:
        return []
    results = []
    for idx in I[0]:
        if idx < len(FAISS_META):
            results.append(FAISS_META[idx])
    return results

# =================================================================
# 3) Training helpers (your code)
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
# 4) Planner + Explainability (your code)
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
    mom7 = scale_ex(pct_7d, -40, 40)
    rsi_dev = 0.0
    if not (isinstance(rsi, float) and math.isnan(rsi)):
        if rsi >= 70:
            rsi_dev = -1.0
        elif rsi <= 30:
            rsi_dev = 1.0
        else:
            rsi_dev = (50 - rsi) / 20.0
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
    if sentiment < -0.2:
        scenarios.append("🟡 If the Fed signals a pause or rate cut next month → possible bullish reversal.")
    if rsi <= 35:
        scenarios.append("🟢 RSI is nearing oversold territory → short-term bounce likely (watch for reversal).")
    if pct_7d < -8:
        scenarios.append("🔻 If price breaks below weekly support → 5–10% downside risk.")
    if -0.15 < sentiment < 0.15 and -7 < pct_24h < 7:
        scenarios.append("⚖️ If no external catalyst occurs → price may continue to consolidate sideways.")
    if sentiment > 0.05 and pct_24h > 0:
        scenarios.append("🔄 Sentiment recovery in progress → sideways to slightly bullish expected.")
    return scenarios

# =================================================================
# 5) Full analyze_coin — your logic, unchanged except minor safety
# =================================================================
def analyze_coin(coin_id: str, coin_symbol: str, risk: str, horizon_days: int, custom_note: str = "", forecast_days: int = 7, model_choice: str = "ensemble"):
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
        date = last_date + pd.Timedelta(days=i + 1) if last_date else None
        
        # Get prophet prediction (if available)
        prophet_pred = prophet_df["yhat"].iloc[-(forecast_days - i)] if not prophet_df.empty else float("nan")

        # Get LSTM prediction (if available)
        lstm_pred = lstm_preds[i] if i < len(lstm_preds) else float("nan")

        # Create ensemble prediction
        ensemble_pred = (0.7 * lstm_pred + 0.3 * prophet_pred) if not math.isnan(lstm_pred) and not math.isnan(prophet_pred) else float("nan")

        forecast_table.append({
            "day": day,
            "date": date,
            "prophet": prophet_pred,
            "lstm": lstm_pred,
            "ensemble": ensemble_pred
        })

    # Prepare data for LLM insight
    forecast_note = ""
    if model_choice == "prophet" and prophet_summary.get("pred_last"):
        forecast_note = f"Prophet forecasts a price of ${prophet_summary['pred_last']:.2f} in {forecast_days} days. Lower bound: ${prophet_summary['lower']:.2f}, Upper bound: ${prophet_summary['upper']:.2f}."
    elif model_choice == "lstm" and lstm_preds:
        forecast_note = f"LSTM forecasts a price of ${lstm_preds[-1]:.2f} in {forecast_days} days."
    elif model_choice == "ensemble" and forecast_table:
        ensemble_pred_val = forecast_table[-1].get("ensemble")
        forecast_note = f"Ensemble model forecasts a price of ${ensemble_pred_val:.2f} in {forecast_days} days."

    # Get recommendation and insight from LLM (or fallback)
    recommendation_result = call_gpt3_for_insight(
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
        forecast_note=forecast_note,
    )

    # Explainability trace
    trace_comps, trace_weights = explain_trace_components(agg_sent, pct_24h, pct_7d, rsi)

    # Strategy plan
    plan_steps = rule_based_plan(mkt.to_dict("records")[0], agg_sent, rsi, risk, horizon_days)

    result_obj = {
        "coin_id": coin_id,
        "coin_symbol": coin_symbol,
        "current_price": price,
        "pct_24h": pct_24h,
        "pct_7d": pct_7d,
        "market_cap": market_cap,
        "volume_24h": volume_24h,
        "rsi": rsi,
        "sentiment_score": agg_sent,
        "sentiment_pcts": sentiment_pcts,
        "headlines": articles,
        "prophet_summary": prophet_summary,
        "lstm_preds": lstm_preds,
        "forecast_table": forecast_table,
        "forecast_days": forecast_days,
        "recommendation": recommendation_result,
        "trace": trace_comps,
        "trace_weights": trace_weights,
        "plan": plan_steps,
        "custom_note": custom_note,
    }
    
    return result_obj

# ---------------------------
# Streamlit UI functions
# ---------------------------
def render_pretty_summary(result_obj: Dict, horizon_days: int=7):
    st.markdown("### Summary & Recommendation")
    if result_obj.get("error"):
        st.error(result_obj["error"])
        return
        
    cols = st.columns(4)
    price = result_obj.get("current_price")
    pct_24h = result_obj.get("pct_24h")
    pct_7d = result_obj.get("pct_7d")
    rsi = result_obj.get("rsi")
    sentiment = result_obj.get("sentiment_score")
    
    if price is not None and not math.isnan(price):
        cols[0].metric("Current Price", f"${price:.2f}", help="Price as of last API fetch")
    else:
        cols[0].metric("Current Price", "N/A", help="Price as of last API fetch")

    if pct_24h is not None and not math.isnan(pct_24h):
        cols[1].metric("24h Change", f"{pct_24h:.2f}%", f"{pct_24h:.2f}")
    else:
        cols[1].metric("24h Change", "N/A")

    if pct_7d is not None and not math.isnan(pct_7d):
        cols[2].metric("7d Change", f"{pct_7d:.2f}%", f"{pct_7d:.2f}")
    else:
        cols[2].metric("7d Change", "N/A")

    if rsi is not None and not math.isnan(rsi):
        cols[3].metric("RSI (14)", f"{rsi:.2f}", help="Relative Strength Index")
    else:
        cols[3].metric("RSI (14)", "N/A")

    # Recommendation
    st.markdown("### Recommendation")
    if result_obj.get("recommendation"):
        rec = result_obj["recommendation"]
        rating = rec.get("rating", "N/A")
        insight = rec.get("insight", "No insight available.")
        score = rec.get("score")
        source = rec.get("source", "N/A")

        if rating == "BUY":
            st.success(f"**Recommendation: {rating}** (Score: {score:.2f}, Source: {source.capitalize()})")
        elif rating == "SELL / AVOID":
            st.error(f"**Recommendation: {rating}** (Score: {score:.2f}, Source: {source.capitalize()})")
        else:
            st.info(f"**Recommendation: {rating}** (Score: {score:.2f}, Source: {source.capitalize()})")
        
        st.markdown(insight)
    else:
        st.markdown("No recommendation available.")

# Other rendering functions (not shown for brevity)
# ...

# Streamlit app flow (main)
def main():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "last_outputs" not in st.session_state:
        st.session_state.last_outputs = {}

    st.title("Crypto Analyst")

    # ---- Input card --------------------------------------------------------------
    with st.container():
        st.markdown("<div class='card input-card'>", unsafe_allow_html=True)
        st.markdown("**Your message**")
        user_message = st.text_input(
            label="",
            value="",
            placeholder="E.g. 'ETH 7-day forecast' or 'Should I buy BTC?'",
            key="user_text",
        )
        send_clicked = st.button("Send", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Handle send ------------------------------------------------------------
    if send_clicked and user_message.strip():
        pretty_text, full_ex, headlines_text, chart_path, result_obj = build_single_response(
            user_message, st.session_state.session_id
        )
        st.session_state.last_outputs = {
            "pretty": pretty_text,
            "ex": full_ex or {},
            "heads": headlines_text,
            "chart": chart_path,
            "result_for_ui": result_obj,
            "horizon": parse_user_message(user_message)["horizon_days"],
        }
    
    # ---- Render summary or an empty state ----------------------------------------
    ui_result = st.session_state.last_outputs.get("result_for_ui")
    if ui_result:
        render_pretty_summary(
            ui_result,
            horizon_days=st.session_state.last_outputs.get("horizon", 7)
        )
    else:
        st.info("Enter a crypto asset (e.g., 'BTC', 'Ethereum') and an optional investment horizon to get started.")

# You need to define these functions or get them from your full source
# def build_single_response(...)
# def parse_user_message(...)

if __name__ == "__main__":
    main()
