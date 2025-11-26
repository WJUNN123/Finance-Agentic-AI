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

import streamlit as st
import os
import google.generativeai as genai
import math
from typing import List, Dict

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
        model = genai.GenerativeModel('gemini-2.0-flash')
        
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
        return False
    # Data scaling for price-based training (different from log-return training)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    X_scaled = scaler_y.transform(X.reshape(-1, 1)).reshape(X.shape)
    
    split = int(len(X_scaled) * 0.9)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y_scaled[:split], y_scaled[split:]
    
    model = build_lstm_model(input_shape=(LSTM_WINDOW, 1))
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=LSTM_BATCH, callbacks=[es], verbose=0)
    
    model.save(os.path.join(MODEL_DIR, f"lstm_{coin_id}.h5"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, f"scaler_lstm_{coin_id}.pkl"))
    
    return True

# =================================================================
# 4) Explainability helpers
# =================================================================
def scale_ex(x, lo, hi):
    """Scale a value from [lo, hi] to [-1, 1] for explainability contribution."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return max(-1.0, min(1.0, ((x - lo) / (hi - lo) * 2 - 1)))
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
    
    # Positive sentiment + price dip
    if sentiment > 0.3 and pct_24h < -2:
        scenarios.append("🟢 Strong sentiment with a minor dip → potential 'buy the dip' opportunity.")
        
    # High negative sentiment
    if sentiment < -0.4:
        scenarios.append("🔴 Major negative sentiment → high probability of further downside in the short term.")
        
    # Oversold/Overbought signals
    if rsi <= 35:
        scenarios.append("🟢 RSI is nearing oversold territory → short-term bounce likely (watch for reversal).")
    elif rsi >= 70:
        scenarios.append("🔴 RSI is heavily overbought → expect a pullback or period of consolidation soon.")

    # Strong weekly trend
    if pct_7d > 20:
        scenarios.append("🟡 Extreme 7-day run-up → profit-taking and volatility are major risks now.")
        
    # Regulatory risk (example - can be improved by checking headlines)
    if "regulat" in st.session_state.last_outputs.get("pretty", ""):
        scenarios.append("⚠️ Regulatory news present → market uncertainty increases; adjust stop-loss orders.")

    return scenarios

# =================================================================
# 5) Message parsing
# =================================================================
def parse_user_message(message: str) -> Dict:
    msg = message.lower()
    coin_id = None
    
    # Try to match coin ID or symbol
    for name, cid in COIN_NAME_TO_ID.items():
        if name in msg:
            coin_id = cid; break
    if coin_id is None:
        for sym, cid in COIN_SYMBOL_TO_ID.items():
            if re.search(r'\b' + re.escape(sym) + r'\b', msg):
                coin_id = cid; break
    
    if coin_id is None:
        coin_id = "bitcoin"  # Default to Bitcoin
        
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
    if m:
        horizon_days = int(m.group(1))
    else:
        if "week" in msg:
            horizon_days = 7
        if "month" in msg:
            horizon_days = 30
            
    return {"coin_id": coin_id, "intent": intent, "horizon_days": horizon_days}

# =================================================================
# 6) Pretty output formatter
# =================================================================
def _pick_insight_line(insight_text: str, label: str, fallback: str = "—") -> str:
    if not insight_text: return fallback
    lines = [ln.strip() for ln in insight_text.splitlines() if ln.strip()]
    
    # Look for **LABEL** or LABEL: lines
    patterns = [
        rf"^\*\*{re.escape(label)}\*\*\s*:?\s*(.+)$",
        rf"^{re.escape(label)}\s*:?\s*(.+)$",
    ]
    for ln in lines:
        for pat in patterns:
            m = re.match(pat, ln, flags=re.IGNORECASE)
            if m:
                # Return the whole line for context
                return ln
    return fallback

def make_pretty_output(result: Dict, comps: Dict, vol: float, mc: float, horizon_days: int) -> str:
    
    # Unpack core data
    coin_id = result["coin_id"]
    coin_symbol = result["coin_symbol"]
    rec = result["recommendation"]
    c24 = result["pct_24h"]
    c7 = result["pct_7d"]
    rsi = result["rsi"]
    price = result["price"]
    
    # General status
    rec_label = rec.get("rating", "HOLD / WAIT")
    rec_emoji = "🟢" if "BUY" in rec_label else ("🔴" if "SELL" in rec_label else "🟡")
    
    top_block = "\n".join([
        f"# {rec_emoji} {rec_label} {coin_id.capitalize()} ({coin_symbol.upper()})",
        f"**Current Price**: ${price:,.2f} USD",
        f"**Analysis Horizon**: {horizon_days} Days",
    ])
    
    # --- Sentiment Block ---
    pcts = result["sentiment_pcts"]
    s_pos = pcts.get("positive", 0.0)
    s_neu = pcts.get("neutral", 0.0)
    s_neg = pcts.get("negative", 0.0)
    
    sentiment_lines = [
        f"**Aggregate Sentiment Score**: {result['agg_sent']:.3f} (Range -1 to +1)",
        f"**News Polarity**: {s_pos:.1f}% Positive | {s_neu:.1f}% Neutral | {s_neg:.1f}% Negative",
        f"**Gemini Sentiment Interpretation**: {_pick_insight_line(rec['insight'], 'Sentiment analysis interpretation')}",
    ]
    sentiment_block = "\n".join(["📰 News & Sentiment Analysis", "", *sentiment_lines])
    
    # --- Explainability / Momentum Block ---
    def comp_line(label, key, meaning_pos, meaning_neg, neutral_hint):
        val = comps.get(key, None)
        if val is None: return f"**{label}**: N/A"
        sign = "🟢" if val >= 0.05 else ("🔴" if val <= -0.05 else "🟡")
        meaning = (meaning_pos if val > 0 else meaning_neg) if abs(val) >= 0.05 else neutral_hint
        score_pct = int(abs(val) * 100)
        return f"**{label}**: {sign} {meaning} ({score_pct}% impact)"

    m24_line = comp_line("24h Momentum", "mom24_component", "Strong gain", "Significant drop", "Sideways movement")
    m7_line = comp_line("7d Momentum", "mom7_component", "Strong weekly uptrend", "Strong weekly downtrend", "Consolidation")
    rsi_line = comp_line("RSI (14) Mean Reversion", "rsi_component", "Oversold bounce likely", "Overbought pullback likely", "Neutral zone (50)")

    risk_block = "\n".join([
        f"⚠️ Risk Factors & Strategy (Score: **{int(round(50 + 50 * comps.get('total_score', 0)))}/100**)",
        f"**Gemini Risk Analysis**: {_pick_insight_line(rec['insight'], 'Risk factors to consider')}",
    ])

    strategy_block = "\n".join([
        "🧠 Strategy Simulation (What-If Scenarios)",
        f"**Gemini Key Catalysts**: {_pick_insight_line(rec['insight'], 'Key catalysts to watch')}",
        "",
        *generate_scenarios(result['agg_sent'], rsi, c24, c7)
    ])
    
    # --- Forecast Block ---
    ft = result["forecast_table"]
    ft_lines = []
    if ft:
        last_pred = ft[-1]
        p_last = last_pred.get("ensemble_val") or last_pred.get("lstm_val") or last_pred.get("prophet_val")
        p_upper = result.get("prophet_summary", {}).get("upper")
        p_lower = result.get("prophet_summary", {}).get("lower")
        
        if p_last is not None:
            forecast_note = f"**{horizon_days}-Day Target**: **${p_last:,.2f}**"
            if p_upper and p_lower and abs(p_upper - p_lower) / p_last > 0.01:
                forecast_note += f" (Range ${p_lower:,.2f} - ${p_upper:,.2f})"
            ft_lines.append(forecast_note)
            
            p_initial = result.get("price")
            if p_initial is not None:
                p_change = ((p_last - p_initial) / p_initial) * 100
                ft_lines.append(f"**Predicted Change**: {p_change:+.2f}%")
    
    ft_text = "\n".join(ft_lines) if ft_lines else "No forecast available."
    
    forecast_block = "\n".join([
        f"🔮 Price Forecast ({horizon_days}-Day)",
        "",
        ft_text,
    ])
    
    output = "\n\n".join([
        top_block, 
        sentiment_block,
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
    if a >= 1_000_000_000: return f"${n/1_000_000_000:.2f}B"
    if a >= 1_000_000: return f"${n/1_000_000:.2f}M"
    return f"${n:,.2f}"

def _rsi_zone(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return "Overbought" if v >= 70 else ("Oversold" if v <= 30 else "Neutral")

def _sentiment_bar(pos, neu, neg, width_blocks=20):
    pb = int(round(width_blocks * (pos/100.0)))
    nb = int(round(width_blocks * (neu/100.0)))
    rb = max(0, width_blocks - pb - nb)
    return "🟩"*pb + "⬜"*(nb) + "🟥"*rb

def render_pretty_summary(ui_result: Dict, horizon_days: int):
    # === FIX START: Defensive check for coin_id ===
    coin_id = ui_result.get("coin_id")
    if not coin_id:
        st.error("Error: Analysis result object is missing the coin identifier. Please try a different query.")
        return
    # === FIX END ===

    # Unpack core data for UI display
    rec = ui_result["recommendation"]
    price = ui_result["price"]
    c24 = ui_result["pct_24h"]
    c7 = ui_result["pct_7d"]
    rsi = ui_result["rsi"]
    mcap = ui_result["market_cap"]
    vol24 = ui_result["volume_24h"]
    pcts = ui_result["sentiment_pcts"]
    vol = ui_result.get("volume_24h", 0)
    mc = ui_result.get("market_cap", 1e12)
    liq_pct = (100.0 * vol / mc) if mc > 0 else 0.0
    
    # Recommendation styling
    rec_label = rec.get("rating", "HOLD / WAIT")
    rec_emoji = "🟢" if "BUY" in rec_label else ("🔴" if "SELL" in rec_label else "🟡")
    rec_color = "#38b55d" if "BUY" in rec_label else ("#e34444" if "SELL" in rec_label else "#e3af3d")
    rec_text = f"Recommendation: {rec_label}"

    st.subheader(f"{coin_id.capitalize()} ({ui_result['coin_symbol'].upper()}) Analysis")
    st.caption(f"Outlook for {horizon_days} days | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.divider()

    # =========================
    # Metrics (4 columns)
    # =========================
    cols = st.columns(4)
    with cols[0]:
        st.metric("Current Price (USD)", _fmt_money(price))
        st.markdown(
            f"<span style='font-size: 0.8rem; color: #aaa'>24h Change: {c24:+.2f}%</span>",
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
        st.metric("7d Change", f"{c7:+.2f}%" if isinstance(c7,(int,float)) else "—")
        st.metric("RSI (14)", f"{rsi:.1f}" if isinstance(rsi,(int,float)) else "—")
    with cols[3]:
        st.write("**Quick tags**")
        chips = []
        if isinstance(c7,(int,float)): chips.append(f"7d: {c7:+.2f}%")
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
        rec_source = rec.get("source", "fallback").capitalize()
        
        st.markdown(f"**Generated by {rec_source}**")
        
        # Display Gemini/Fallback insight
        if rec_source == "Gemini":
            st.markdown(rec.get("insight", "No detailed insight available."))
        else:
            st.info(rec.get("insight", "No detailed insight available."))
            
        # Add a placeholder/padding to use up amount of vertical space to fully occupy the column
        for _ in range(15): st.write("")


    with sidebar_col:
        # === SENTIMENT SECTION ===
        st.subheader("📰 Sentiment Score")
        pos_pct = float(pcts.get("positive", 0.0))
        neu_pct = float(pcts.get("neutral", 0.0))
        neg_pct = float(pcts.get("negative", 0.0))
        
        st.markdown(f"**Score**: **{ui_result['agg_sent']:.3f}** (Range -1 to +1)")
        st.markdown(_sentiment_bar(pos_pct, neu_pct, neg_pct))
        st.caption(f"Positive: {pos_pct:.1f}% | Neutral: {neu_pct:.1f}% | Negative: {neg_pct:.1f}%")
        
        # === MOMENTUM & RSI SECTION ===
        st.subheader("📈 Momentum & RSI")
        
        # 24h momentum
        if isinstance(c24,(int,float)):
            if c24 > 2: st.write(f"• **24h**: +{c24:.2f}% (Strong)")
            elif c24 > 0: st.write(f"• **24h**: +{c24:.2f}% (Slight gain)")
            elif c24 > -2: st.write(f"• **24h**: {c24:+.2f}% (Sideways)")
            else: st.write(f"• **24h**: {c24:+.2f}% (Declining)")
        else:
            st.write("• **24h**: Data unavailable")

        # 7d momentum
        if isinstance(c7,(int,float)):
            if c7 > 5: st.write(f"• **7d**: +{c7:.2f}% (Strong trend)")
            elif c7 > 0: st.write(f"• **7d**: +{c7:.2f}% (Weekly gain)")
            elif c7 > -5: st.write(f"• **7d**: {c7:+.2f}% (Consolidation)")
            else: st.write(f"• **7d**: {c7:+.2f}% (Decline)")
        else:
            st.write("• **7d**: Data unavailable")

        # RSI
        if isinstance(rsi,(int,float)):
            zone = _rsi_zone(rsi)
            st.write(f"• **RSI**: {rsi:.1f} ({zone})")
        else:
            st.write("• **RSI**: Data unavailable")
            
        # Adding a large amount of vertical space to fully occupy the column
        for _ in range(15): st.write("")


    # =========================
    # Tabs for detail
    # =========================
    tab1, tab2, tab3, tab4 = st.tabs(["Chart & Forecast", "Explainability Trace", "Headlines", "Memory (LTM)"])
    
    # --- Tab 1: Chart & Forecast ---
    with tab1:
        st.subheader(f"Price History & {horizon_days}-Day Forecast")
        df_hist = ui_result.get("history")
        df_prophet = ui_result.get("prophet_df")
        lstm_preds = ui_result.get("lstm_preds")
        
        # --- Forecast Table ---
        st.markdown("#### Forecast Summary")
        ft_data = ui_result.get("forecast_table", [])
        if ft_data:
            ft_df = pd.DataFrame(ft_data)
            ft_df["Date"] = ft_df["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
            ft_df.rename(columns={"prophet_val": "Prophet", "lstm_val": "LSTM", "ensemble_val": "Ensemble"}, inplace=True)
            st.dataframe(ft_df, use_container_width=True, hide_index=True)
        else:
            st.caption("_No forecast data available._")
            
        st.markdown("#### Interactive Price Chart")
        
        # Altair charting logic
        if df_hist is not None and not df_hist.empty:
            df_hist_chart = df_hist.reset_index().rename(columns={"index": "Date", "price": "History"})
            df_forecast = pd.DataFrame()
            
            # Use ensemble forecast if available, otherwise just LSTM/Prophet
            forecast_series = None
            if lstm_preds and df_prophet is not None and not df_prophet.empty:
                prophet_vals = df_prophet["yhat"].tail(len(lstm_preds)).values
                ensemble_vals = 0.7 * np.array(lstm_preds) + 0.3 * prophet_vals
                forecast_series = pd.Series(ensemble_vals, name="Forecast")
            elif lstm_preds:
                forecast_series = pd.Series(lstm_preds, name="Forecast")
            elif df_prophet is not None and not df_prophet.empty:
                forecast_series = df_prophet["yhat"].tail(horizon_days).reset_index(drop=True).rename("Forecast")

            if forecast_series is not None and not forecast_series.empty:
                last_dt = df_hist.index[-1]
                future_index = [last_dt + pd.Timedelta(days=i+1) for i in range(len(forecast_series))]
                df_forecast = pd.DataFrame({"Date": future_index, "Forecast": forecast_series.values})

            plot_df = pd.concat([df_hist_chart, df_forecast], ignore_index=True)
            plot_df["Date"] = pd.to_datetime(plot_df["Date"])
            
            import altair as alt
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
            
            mc_mean = ui_result.get("mc_mean") or []
            mc_lo = ui_result.get("mc_lo") or []
            mc_hi = ui_result.get("mc_hi") or []
            
            chart = (lines + points)
            try:
                if mc_mean and len(mc_mean) == df_forecast.shape[0]:
                    band_df = pd.DataFrame({
                        "Date": pd.to_datetime(df_forecast["Date"]),
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


    # --- Tab 2: Explainability Trace ---
    with tab2:
        st.subheader("Decision Breakdown: Rule-Based Components")
        st.markdown("**This shows the component weights of the internal score (before Gemini's input).**")
        
        comps = ui_result.get("explain_comps", {})
        friendly_ex_text = st.session_state.last_outputs.get("ex", {}).get("explain_summary", "N/A")
        
        if comps:
            st.json(comps)
        else:
            st.warning("Explainability trace is currently empty. Re-run the query.")
            
        st.markdown("#### Component Interpretation")
        st.markdown(friendly_ex_text)

    # --- Tab 3: Headlines ---
    with tab3:
        st.subheader("Top Headlines & Sentiment Analysis")
        dfh = ui_result.get("sentiment_table")
        if dfh is not None and not dfh.empty:
            df_display = dfh[["text", "label", "score"]].copy()
            df_display.rename(columns={"text": "Headline", "label": "FinBERT Label", "score": "Confidence"}, inplace=True)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.caption("_No news or sentiment analysis data available._")

    # --- Tab 4: Memory ---
    with tab4:
        st.subheader("Conversation History (Short-Term Memory)")
        history = load_recent_session(st.session_state.session_id)
        if history:
            for msg in reversed(history):
                role = "user" if msg["role"] == "user" else "assistant"
                is_full_ex = msg["meta"].get("explain_full", False)
                
                if role == "assistant" and is_full_ex:
                    # Skip the full JSON explanation stored in memory
                    continue
                
                with st.chat_message(role):
                    if role == "assistant":
                        if msg["meta"].get("chart"):
                            st.caption(f"Chart saved: {msg['meta']['chart']}")
                        if msg["meta"].get("explain_summary"):
                            st.caption(f"Explanation: {msg['meta']['explain_summary'][:80]}...")
                            
                    st.markdown(msg["message"])
        else:
            st.caption("_No conversation history for this session._")
            
        st.markdown("#### Long-Term Memory (FAISS Retrieval)")
        if EMB_AVAILABLE:
            st.caption("FAISS is initialized. Last query results (top 5 similar items):")
            ltm_results = retrieve_similar(ui_result.get("user_message", "crypto analysis"), k=5)
            if ltm_results:
                for idx, res in enumerate(ltm_results):
                    st.text(f"{idx+1}. Score: {res.get('meta', {}).get('score', 'N/A')}\n   Text: {res['text'][:100]}...")
            else:
                st.caption("_No similar items found in Long-Term Memory._")
        else:
            st.caption("_FAISS/SentenceTransformers not installed. Long-Term Memory inactive._")


# =================================================================
# 8) Build response (returns structured `result` for the UI)
# =================================================================
def build_single_response(user_message: str, session_id: str):
    parsed = parse_user_message(user_message)
    coin_id = parsed["coin_id"]
    horizon_days = parsed["horizon_days"]
    coin_symbol = next((c["symbol"] for c in DEFAULT_COINS if c["id"] == coin_id), coin_id[:4])
    
    save_conversation(session_id, "user", user_message)

    # Core analysis with GPT-3 insights
    result = analyze_and_forecast(coin_id, coin_symbol, horizon_days, user_message)
    
    # Check for error immediately after analysis and exit gracefully before accessing other keys
    if "error" in result:
        # Save error message to short-term memory
        error_msg = f"Error during analysis for {coin_id}: {result['error']}"
        save_conversation(session_id, "assistant", error_msg, {})
        # Return error details to the Streamlit loop
        return error_msg, {}, "", None, result
    
    # Get components for structured output
    agg_sent = result.get("agg_sent", 0.0)
    c24 = result.get("pct_24h", float("nan"))
    c7 = result.get("pct_7d", float("nan"))
    rsi = result.get("rsi", float("nan"))
    vol = result.get("volume_24h", 0.0)
    mc = result.get("market_cap", 1e12)
    comps, weights = explain_trace_components(agg_sent, c24, c7, rsi)
    result["explain_comps"] = comps

    # Format output for display/memory
    pretty = make_pretty_output(result, comps, vol, mc, horizon_days)
    ex = {"explain_summary": "", "full_explain": comps} # Simple object for full memory storage
    
    # Friendly explanation summary for memory
    def _describe_component(label, val, meaning_pos, meaning_neg):
        sign = "Positive" if val >= 0.05 else ("Negative" if val <= -0.05 else "Neutral")
        meaning = meaning_pos if val > 0 else meaning_neg
        return f"{label}: {sign} ({abs(val):.2f}) -> {meaning}"

    friendly_ex = [
        _describe_component("Sentiment", comps.get("sentiment_component", 0), "positive newsflow", "negative newsflow"),
    ]
    if "mom24_component" in comps:
        friendly_ex.append(_describe_component("Mom24", comps["mom24_component"], "strong 24h momentum", "24h momentum weak"))
    if "mom7_component" in comps:
        friendly_ex.append(_describe_component("Mom7", comps["mom7_component"], "7d momentum uptrend", "7d momentum flat/weak"))
    if "rsi_component" in comps:
        friendly_ex.append(_describe_component("RSI", comps["rsi_component"], "mean reversion expected (oversold)", "mean reversion expected (overbought)"))
        
    friendly_ex_text = "\n".join(friendly_ex)
    
    # Headlines text
    if not result.get("sentiment_table", pd.DataFrame()).empty:
        dfh = result["sentiment_table"]
        headlines_text = "\n".join([f"{r['text']} -> {r['label']} ({r['score']:.2f})" for _, r in dfh.head(6).iterrows()])
    else:
        headlines_text = "\n".join([a.get("title", "") for a in result.get("articles", [])[:6]])

    # Chart file
    chart_path = plot_history_forecasts_to_file(
        result.get("history", pd.DataFrame()),
        result.get("prophet_df", pd.DataFrame()),
        result.get("lstm_preds", []),
        coin_id,
    )
    
    # Save to short-term memory
    save_conversation(session_id, "assistant", pretty, {"chart": chart_path, "explain_summary": friendly_ex_text})
    save_conversation(session_id, "assistant", json.dumps(ex), {"explain_full": True})
    
    # Optionally save to long-term memory
    if agg_sent != 0.0 and result.get("price") is not None:
        ltm_meta = {
            "coin": coin_id, 
            "price": result["price"], 
            "score": comps.get("total_score", 0.0),
            "recommendation": result["recommendation"].get("rating", "HOLD")
        }
        add_long_term_item(user_message + " -> " + pretty[:200], ltm_meta)

    return pretty, ex, headlines_text, chart_path, result


def analyze_and_forecast(coin_id: str, coin_symbol: str, horizon_days: int, user_message: str) -> Dict:
    # 1. Fetch market data
    try:
        market_df = coingecko_market([coin_id])
    except Exception as e:
        return {"error": f"Failed to fetch market data from CoinGecko: {e}"}
        
    if market_df.empty:
        return {"error": f"Could not find market data for coin ID: {coin_id}"}
        
    coin_data = market_df.iloc[0]
    price = float(coin_data.get("current_price", float("nan")))
    pct_24h = float(coin_data.get("price_change_percentage_24h", float("nan")))
    pct_7d = float(coin_data.get("price_change_percentage_7d_in_currency", float("nan")))
    market_cap = float(coin_data.get("market_cap", float("nan")))
    volume_24h = float(coin_data.get("total_volume", float("nan")))
    
    # 2. Fetch history and compute RSI
    hist_df = coingecko_chart(coin_id, days=180)
    rsi = compute_rsi(hist_df["price"] if not hist_df.empty else None, period=14)
    
    # 3. Fetch news and run sentiment analysis
    articles = fetch_rss_articles(coin_id, limit_per_feed=20)
    headlines = [a["title"] for a in articles]
    sentiment_analyses = run_finbert(headlines)
    agg_sent, sentiment_table = sentiment_score(sentiment_analyses)
    sentiment_pcts = sentiment_percentages(sentiment_analyses)
    
    # 4. Run forecasts (use ensemble for Gemini context)
    prophet_df, prophet_summary = forecast_with_prophet(hist_df["price"] if (not hist_df.empty and "price" in hist_df.columns) else pd.Series(), days=horizon_days)
    lstm_preds = train_and_forecast_lstm(hist_df["price"] if (not hist_df.empty and "price" in hist_df.columns) else pd.Series(), horizon=horizon_days, window=LSTM_WINDOW, epochs=LSTM_EPOCHS)

    # Determine forecast note for Gemini
    forecast_note = ""
    last_prophet = prophet_summary.get("pred_last")
    if lstm_preds:
        last_lstm = lstm_preds[-1]
        if last_prophet:
            ensemble_pred = 0.7 * last_lstm + 0.3 * last_prophet
            forecast_note = f"Ensemble target is ${ensemble_pred:.2f} in {horizon_days} days. LSTM: ${last_lstm:.2f}, Prophet: ${last_prophet:.2f}."
        else:
            forecast_note = f"LSTM model predicts a price of ${last_lstm:.2f} in {horizon_days} days."
    elif last_prophet:
        forecast_note = f"Prophet model predicts a price range of ${prophet_summary['lower']:.2f} to ${prophet_summary['upper']:.2f} with a central prediction of ${last_prophet:.2f} in {horizon_days} days."

    # 5. Call Gemini for insight and recommendation
    risk_level = "Medium" # Simplification for demo
    recommendation = call_gpt3_for_insight(
        coin_id=coin_id,
        coin_symbol=coin_symbol,
        sentiment=agg_sent,
        pct_24h=pct_24h,
        pct_7d=pct_7d,
        rsi=rsi,
        price=price,
        market_cap=market_cap,
        volume_24h=volume_24h,
        risk=risk_level,
        horizon_days=horizon_days,
        top_headlines=headlines[:5],
        forecast_note=forecast_note
    )

    # 6. Build forecast table for UI
    forecast_table = []
    last_date = hist_df.index[-1] if (hist_df is not None and not hist_df.empty) else None
    for i in range(horizon_days):
        day = i + 1
        date = (last_date + pd.Timedelta(days=day)) if last_date is not None else None
        
        prophet_val = None
        try:
            if (prophet_df is not None) and (not prophet_df.empty):
                future_rows = prophet_df.tail(horizon_days).reset_index(drop=True)
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
            
        forecast_table.append({
            "Day": day,
            "Date": date,
            "prophet_val": prophet_val,
            "lstm_val": lstm_val,
            "ensemble_val": ensemble_val,
        })
        
    # 7. Monte-Carlo uncertainty band (simplistic for log-returns model)
    mc_mean, mc_lo, mc_hi = [], [], []
    try:
        if hist_df is not None and not hist_df.empty and "price" in hist_df.columns and lstm_preds:
            hist_prices = hist_df["price"].astype(float)
            p0 = float(hist_prices.iloc[-1])
            logrets = np.log(hist_prices).diff().dropna()
            
            # Estimate volatility
            if not logrets.empty:
                sig = float(logrets.ewm(span=20, adjust=False).std().iloc[-1])
            else:
                sig = 0.0
                
            sig = 0.0 if (isinstance(sig, float) and (math.isnan(sig) or sig < 1e-9)) else sig
            
            # Estimate drift (average return from LSTM predictions)
            drift = np.diff(np.log([p0] + list(lstm_preds)))
            horizon = len(lstm_preds)
            if len(drift) < horizon and len(drift) > 0:
                 drift = np.pad(drift, (0, horizon - len(drift)), mode="edge")

            rng = np.random.default_rng(42)
            n_paths = 50 
            sims = []
            
            for _ in range(n_paths):
                # Add noise component
                noise = rng.normal(0.0, sig, size=horizon) * 0.6 
                rets = drift + noise
                path = p0 * np.exp(np.cumsum(rets))
                sims.append(path)
                
            sims = np.array(sims)
            mc_mean = sims.mean(axis=0).tolist()
            mc_lo = np.percentile(sims, 25, axis=0).tolist()
            mc_hi = np.percentile(sims, 75, axis=0).tolist()
    except Exception:
        mc_mean, mc_lo, mc_hi = [], [], []


    # 8. Compile final result object
    return {
        "coin_id": coin_id,
        "coin_symbol": coin_symbol,
        "price": price,
        "pct_24h": pct_24h,
        "pct_7d": pct_7d,
        "market_cap": market_cap,
        "volume_24h": volume_24h,
        "rsi": rsi,
        "agg_sent": agg_sent,
        "sentiment_pcts": sentiment_pcts,
        "sentiment_table": sentiment_table,
        "articles": articles,
        "history": hist_df,
        "prophet_df": prophet_df,
        "prophet_summary": prophet_summary,
        "lstm_preds": lstm_preds,
        "forecast_table": forecast_table,
        "recommendation": recommendation,
        "user_message": user_message,
        "mc_mean": mc_mean,
        "mc_lo": mc_lo,
        "mc_hi": mc_hi,
    }


# =================================================================
# 9) STREAMLIT APP (Summary-only UI, polished with Send below input)
# =================================================================

# --- Initialize session state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.last_outputs = {}

st.markdown("""
<style>
/* Overall spacing */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.4rem;
}
/* Input card styling */
.input-card {
    border: 1px solid #1f2a38;
    border-radius: 12px;
    padding: 15px;
    background-color: #141c27;
}
/* Style Streamlit text input box */
div[data-testid="stTextInput"] > div > div > input {
    border-radius: 8px;
    background-color: #0d1117; 
    border: 1px solid #1f2a38;
}
/* Custom chips/prompts styling */
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 15px;
}
.chips span {
    background-color: #1b2332;
    color: #a0a6b0;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: background-color 0.2s;
    border: 1px solid #1b2332;
}
.chips span:hover {
    background-color: #263346;
    color: #eaf0ff;
}

</style>
""", unsafe_allow_html=True)

# ---- Title and context -------------------------------------------------------
st.title("Crypto Analyst")
st.caption("AI-powered technical, sentiment, and fundamental analysis for cryptocurrencies.")

# ---- Quick prompts (clickable chips) -----------------------------------------
prompts = [
    "BTC 7-day forecast",
    "Should I buy ETH?",
    "What is the sentiment for SOL?",
    "XRP 30-day price prediction",
]
html = "<div class='chips'>" + "".join(
    [f"<span onclick=\"window.parent.postMessage({{'type':'streamlit:setComponentValue','value':'{p}'}}, '*')\">{p}</span>" for p in prompts]
) + "</div>"
st.markdown(html, unsafe_allow_html=True)

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
    # Send button moved BELOW input
    send_clicked = st.button("Send", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Handle send -------------------------------------------------------------
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
    # Check if a Gemini error occurred
    if "error" in ui_result:
        st.error(f"Error: {ui_result['error']}")
    else:
        # Check if fallback occurred and warn the user (optional, as the UI handles it)
        if ui_result.get("recommendation", {}).get("source") == "fallback":
             st.warning("Could not connect to Gemini API. Analysis is based on rule-based fallback logic.")

        render_pretty_summary(
            ui_result,
            horizon_days=st.session_state.last_outputs.get("horizon", 7)
        )
else:
    st.info("Start by asking a question about a cryptocurrency, e.g., 'What is the 7-day forecast for BNB?'")
