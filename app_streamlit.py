# app_streamlit.py — FULL conversion from Gradio to Streamlit
# ---------------------------------------------------------------------------------
# UPDATES:
# 1. Sentiment: FinBERT -> Twitter-RoBERTa
# 2. Ensemble: Simple Average -> XGBoost Stacking (Prophet + LSTM)
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

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Prophet
from prophet import Prophet

# TensorFlow for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Optional: sentence-transformers + faiss for embeddings and retrieval
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMB_AVAILABLE = True
except Exception:
    EMB_AVAILABLE = False

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="Crypto Analyst — Streamlit", layout="wide")

# ---------------------------
# Configuration
# ---------------------------
# CHANGE 1: Switched to Twitter-RoBERTa
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

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

# Hugging Face token (optional)
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
    if logrets is None or len(logrets.dropna()) < 5:
        return float("nan")
    return float(logrets.ewm(span=span, adjust=False).std().iloc[-1])

def compound_from_returns(p0: float, rets: np.ndarray) -> np.ndarray:
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
    seen = set()
    deduped = []
    for it in sorted(items, key=lambda x: x["published_ts"], reverse=True):
        if it["title"] not in seen:
            seen.add(it["title"])
            deduped.append(it)
    return deduped[:50]

# ---------------------------
# Sentiment (Twitter-RoBERTa)
# ---------------------------
SENTIMENT_PIPE = None

def load_sentiment_pipeline():
    global SENTIMENT_PIPE
    SENTIMENT_PIPE = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=PIPELINE_DEVICE
    )
    return SENTIMENT_PIPE

def run_sentiment_analysis(headlines: List[str]) -> List[Dict]:
    global SENTIMENT_PIPE
    if not headlines:
        return []
    if SENTIMENT_PIPE is None:
        try:
            SENTIMENT_PIPE = load_sentiment_pipeline()
        except Exception as e:
            raise RuntimeError(f"Failed to load Sentiment Model: {e}")
            
    # Twitter-RoBERTa outputs: LABEL_0 (Negative), LABEL_1 (Neutral), LABEL_2 (Positive)
    preds = SENTIMENT_PIPE(headlines, truncation=True, max_length=512)
    
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    
    out = []
    for h, p in zip(headlines, preds):
        raw_label = p.get("label", "")
        label = label_map.get(raw_label, "neutral") # Map RoBERTa labels
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
        return pd.DataFrame(), {"error": "Not enough history for Prophet"}

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
# LSTM (Modified to support Stacking)
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

def train_and_forecast_lstm_full(price_series: pd.Series,
                                 horizon: int = 7,
                                 window: int = LSTM_WINDOW,
                                 epochs: int = LSTM_EPOCHS):
    """
    Returns (historical_predictions, future_predictions) for XGBoost stacking.
    historical_predictions: aligned with the end of the input series.
    """
    if price_series is None or len(price_series) < window + 20:
        return np.array([]), []

    s = price_series.astype(float).copy()
    logp = np.log(s)
    ret = logp.diff()

    df = pd.DataFrame({"price": s, "ret": ret})
    df["ma7_dist"]  = (s - s.rolling(7).mean()) / s
    df["ma14_dist"] = (s - s.rolling(14).mean()) / s
    df["rsi14"] = rsi_series(s, 14)

    df = df.dropna().copy()
    if df.empty or len(df) < window + horizon:
        return np.array([]), []

    # Features
    feats = df[["ret", "ma7_dist", "ma14_dist", "rsi14"]].values
    y_ret = df["ret"].shift(-1).dropna().values
    feats = feats[:-1, :]

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(feats)

    # Sliding window
    X, y = [], []
    for i in range(window, len(X_scaled)):
        X.append(X_scaled[i - window:i, :])
        y.append(y_ret[i])
    X, y = np.array(X), np.array(y)

    # Train on Full History available (for demo purposes)
    model = build_lstm_model(input_shape=(window, X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=LSTM_BATCH, verbose=0)

    # 1. Generate IN-SAMPLE predictions (for stacking training)
    # We predict the 'ret' for each window, then compound to get price
    in_sample_pred_rets = model.predict(X, verbose=0).flatten()
    
    # Align in-sample predictions with timestamps
    # X[i] uses data up to index (window+i-1) to predict index (window+i)
    # The original 'df' index starts after dropna.
    # We need to map these returns back to prices.
    
    # Recover prices from returns: Price[t] = Price[t-1] * exp(pred_ret)
    # The 'y' array corresponds to returns at df index [window:]
    valid_start_idx = window 
    base_prices = df["price"].values[valid_start_idx : valid_start_idx + len(in_sample_pred_rets)]
    # Note: LSTM predicts "next day return". 
    # So pred[0] is return for day (window+1). Base price is day (window).
    
    in_sample_prices = base_prices * np.exp(in_sample_pred_rets)
    
    # 2. Generate OUT-OF-SAMPLE (Future) predictions
    last_window = X_scaled[-window:, :].copy()
    future_rets = []
    
    curr_window = last_window
    for _ in range(horizon):
        x_in = curr_window.reshape((1, window, X.shape[2]))
        pred_ret = float(model.predict(x_in, verbose=0)[0][0])
        future_rets.append(pred_ret)
        
        # update window
        feat_next = curr_window[-1, :].copy()
        feat_next[0] = pred_ret 
        feat_next = feat_next.reshape(1, -1)
        curr_window = np.vstack([curr_window[1:], feat_next])

    p0 = float(s.iloc[-1])
    future_path = compound_from_returns(p0, np.array(future_rets)).tolist()
    
    return in_sample_prices, future_path

# ---------------------------
# CHANGE 2: XGBoost Stacking
# ---------------------------
def run_xgboost_stacking(hist_df, prophet_df, lstm_in_sample, lstm_future, horizon_days):
    """
    Trains XGBoost to combine Prophet and LSTM outputs based on historical accuracy.
    """
    if not XGB_AVAILABLE:
        # Fallback if XGBoost not installed
        return [0.5 * l + 0.5 * p for l, p in zip(lstm_future, prophet_df["yhat"].tail(horizon_days))]

    # 1. Prepare Training Data for Stacking
    # We need overlapping period where we have: Actual Price, Prophet Pred, LSTM Pred
    
    # Prophet in-sample
    # Prophet df covers full history. We need to align with LSTM in-sample.
    # LSTM in-sample length is len(in_sample_prices). It ends at the last historical point.
    
    overlap_len = len(lstm_in_sample)
    if overlap_len < 10:
        # Not enough overlap to train stacker, fallback to average
        p_fut = prophet_df["yhat"].tail(horizon_days).values
        return [0.7 * l + 0.3 * p for l, p in zip(lstm_future, p_fut)]
    
    # Actual prices for the overlap period (the end of history)
    y_actual = hist_df["price"].tail(overlap_len).values
    
    # Prophet preds for overlap
    # Prophet "yhat" aligns with "ds". We take the last 'overlap_len' records.
    p_hist = prophet_df["yhat"].iloc[-overlap_len:].values
    
    # LSTM preds for overlap
    l_hist = lstm_in_sample
    
    # RSI for overlap (as an extra feature for the meta-learner)
    rsi_hist = rsi_series(hist_df["price"], 14).tail(overlap_len).fillna(50).values
    
    # Construct Feature Matrix (X) and Target (y)
    X_train = pd.DataFrame({
        "prophet": p_hist,
        "lstm": l_hist,
        "rsi": rsi_hist
    })
    
    # Train XGBoost Regressor
    # Objective: Minimize error in price prediction
    model_xgb = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective='reg:squarederror'
    )
    model_xgb.fit(X_train, y_actual)
    
    # 2. Predict Future
    p_future = prophet_df["yhat"].tail(horizon_days).values
    l_future = np.array(lstm_future)
    # Assume RSI stays roughly constant or drift it (simplification for stacking feature)
    last_rsi = rsi_hist[-1]
    rsi_future = np.full(horizon_days, last_rsi)
    
    X_future = pd.DataFrame({
        "prophet": p_future,
        "lstm": l_future,
        "rsi": rsi_future
    })
    
    final_preds = model_xgb.predict(X_future)
    return final_preds.tolist()

# ---------------------------
# Chart
# ---------------------------
def plot_history_forecasts_to_file(hist: pd.DataFrame, prophet_forecast: pd.DataFrame, lstm_preds: List[float], ensemble_preds: List[float], coin_id: str) -> str:
    if hist is None or hist.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4))
    
    # Plot History
    ax.plot(hist.index, hist["price"], label="History", linewidth=2, color="black")
    
    # Plot Prophet (Optional visual context)
    if prophet_forecast is not None and not prophet_forecast.empty:
        try:
            fc_idx = pd.to_datetime(prophet_forecast["ds"])
            ax.plot(fc_idx, prophet_forecast["yhat"], linestyle="--", linewidth=1, label="Prophet", alpha=0.5)
        except Exception:
            pass
            
    # Plot LSTM (Optional visual context)
    if lstm_preds:
        last_dt = hist.index[-1]
        future_index = [last_dt + pd.Timedelta(days=i+1) for i in range(len(lstm_preds))]
        ax.plot(future_index, lstm_preds, linestyle=":", linewidth=1, label="LSTM", alpha=0.5)
        
    # Plot XGBoost Ensemble (Main Forecast)
    if ensemble_preds:
        last_dt = hist.index[-1]
        future_index = [last_dt + pd.Timedelta(days=i+1) for i in range(len(ensemble_preds))]
        ax.plot(future_index, ensemble_preds, linestyle="-", linewidth=2.5, color="#ff4b4b", label="XGBoost Ensemble")
        ax.annotate(f"${ensemble_preds[-1]:.2f}", (future_index[-1], ensemble_preds[-1]), textcoords="offset points", xytext=(0, -15), ha='center', weight="bold", color="#ff4b4b")

    ax.set_title(f"{coin_id.capitalize()} — XGBoost Stacked Forecast", fontsize=12, fontweight="bold")
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
        "insight": f"Based on sentiment {sentiment:.2f}, the recommendation is {recommendation}.",
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
    gemini_api_key = st.secrets.get("gemini", {}).get("api_key", None)
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
        except Exception as e:
            return recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)
    else:
        return recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)
    
    headlines_context = ""
    if top_headlines:
        headlines_context = f"\n\nTop recent headlines:\n" + "\n".join([f"- {h}" for h in top_headlines[:5]])

    def safe_format(val, default="N/A", format_str="{:.2f}"):
        if val is None or (isinstance(val, float) and math.isnan(val)): return default
        return format_str.format(val)

    rsi_zone = "N/A"
    if not (isinstance(rsi, float) and math.isnan(rsi)):
        rsi_zone = "Overbought" if rsi >= 70 else ("Oversold" if rsi <= 30 else "Neutral")

    prompt = f"""You are an expert cryptocurrency analyst. Analyze the following data for {coin_id.upper()} ({coin_symbol.upper()}).
    
    MARKET DATA:
    - Current Price: ${safe_format(price)}
    - Market Cap: ${safe_format(market_cap, format_str="{:,.0f}")}
    - 24h Change: {safe_format(pct_24h)}%
    - 7d Change: {safe_format(pct_7d)}%
    - RSI (14): {safe_format(rsi)} ({rsi_zone})
    
    SENTIMENT ANALYSIS (Twitter-RoBERTa):
    - Score: {sentiment:.3f} (-1.0 to +1.0)
    
    FORECAST:
    {forecast_note}
    
    Headlines:{headlines_context}
    
    Provide a professional BUY/SELL/HOLD recommendation and brief rationale.
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        rating = "HOLD / WAIT"
        if "buy" in text.lower() and "sell" not in text.lower(): rating = "BUY"
        elif "sell" in text.lower() and "buy" not in text.lower(): rating = "SELL / AVOID"
        
        return {"rating": rating, "score": 0.0, "insight": text, "source": "gemini"} 
    except Exception:
        return recommend_and_insight(sentiment, pct_24h, pct_7d, rsi, risk, horizon_days)

# =================================================================
# Explainability & Memory (Unchanged logic)
# =================================================================
def scale_ex(x, lo, hi):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)): return 0.0
        return max(-1.0, min(1.0, ((x - lo) / (hi - lo) * 2 - 1)))
    except Exception: return 0.0

def explain_trace_components(agg_sent, pct_24h, pct_7d, rsi):
    mom24 = scale_ex(pct_24h, -15, 15)
    mom7 = scale_ex(pct_7d, -40, 40)
    rsi_dev = 0.0
    if not (isinstance(rsi, float) and math.isnan(rsi)):
        if rsi >= 70: rsi_dev = -1.0
        elif rsi <= 30: rsi_dev = 1.0
        else: rsi_dev = (50 - rsi) / 20.0  
    weights = {"w_sent": 0.45, "w_m24": 0.2, "w_m7": 0.2, "w_rsi": 0.15}
    comp = {
        "sentiment_component": float(weights["w_sent"] * agg_sent),
        "mom24_component": float(weights["w_m24"] * mom24),
        "mom7_component": float(weights["w_m7"] * mom7),
        "rsi_component": float(weights["w_rsi"] * rsi_dev),
    }
    comp["total_score"] = sum(comp.values())
    return comp, weights

def generate_scenarios(sentiment, rsi, pct_24h, pct_7d):
    scenarios = []
    if sentiment > 0.3 and pct_24h < -2: scenarios.append("🟢 Strong sentiment + price dip → Potential entry.")
    if sentiment < -0.4: scenarios.append("🔴 Negative sentiment → Risk of downside.")
    if rsi <= 35: scenarios.append("🟢 RSI Oversold → Bounce likely.")
    elif rsi >= 70: scenarios.append("🔴 RSI Overbought → Pullback likely.")
    return scenarios

# =================================================================
# Memory DB
# =================================================================
def init_memory_db(path=MEM_DB):
    con = sqlite3.connect(path, check_same_thread=False)
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, message TEXT, meta_json TEXT, ts TIMESTAMP)")
    con.commit()
    return con
MEM_CONN = init_memory_db()

def save_conversation(session_id, role, message, meta=None):
    try:
        cur = MEM_CONN.cursor()
        cur.execute("INSERT INTO conversations (session_id, role, message, meta_json, ts) VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, message, json.dumps(meta or {}), datetime.utcnow().isoformat()))
        MEM_CONN.commit()
    except Exception: pass

def load_recent_session(session_id, limit=20):
    try:
        cur = MEM_CONN.cursor()
        cur.execute("SELECT role, message, meta_json, ts FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit))
        return [{"role": r[0], "message": r[1], "meta": json.loads(r[2]), "ts": r[3]} for r in reversed(cur.fetchall())]
    except Exception: return []

# =================================================================
# Main Logic
# =================================================================
def analyze_and_forecast(coin_id: str, coin_symbol: str, horizon_days: int, user_message: str) -> Dict:
    # 1. Market Data
    try:
        market_df = coingecko_market([coin_id])
    except Exception as e:
        return {"error": f"API Error: {e}"}
    
    if market_df.empty: return {"error": f"Coin not found: {coin_id}"}
    
    coin_data = market_df.iloc[0]
    
    # 2. History
    hist_df = coingecko_chart(coin_id, days=180)
    rsi = compute_rsi(hist_df["price"] if not hist_df.empty else None, 14)
    
    # 3. Sentiment (Twitter-RoBERTa)
    articles = fetch_rss_articles(coin_id)
    headlines = [a["title"] for a in articles]
    sentiment_analyses = run_sentiment_analysis(headlines) # New function
    agg_sent, sentiment_table = sentiment_score(sentiment_analyses)
    sentiment_pcts = sentiment_percentages(sentiment_analyses)
    
    # 4. Forecasts (Stacking)
    prophet_df, prophet_summary = forecast_with_prophet(hist_df["price"] if not hist_df.empty else pd.Series(), days=horizon_days)
    
    # LSTM returns (In-Sample, Future)
    lstm_hist, lstm_future = train_and_forecast_lstm_full(hist_df["price"] if not hist_df.empty else pd.Series(), horizon=horizon_days)
    
    # XGBoost Stacking
    if lstm_future and not prophet_df.empty:
        ensemble_preds = run_xgboost_stacking(hist_df, prophet_df, lstm_hist, lstm_future, horizon_days)
        forecast_note = f"XGBoost Ensemble predicts ${ensemble_preds[-1]:.2f} in {horizon_days} days (combining Prophet & LSTM)."
    else:
        ensemble_preds = []
        forecast_note = "Forecasting failed due to insufficient history."

    # 5. Gemini Insight
    rec = call_gpt3_for_insight(
        coin_id, coin_symbol, agg_sent, 
        coin_data.get("price_change_percentage_24h"), 
        coin_data.get("price_change_percentage_7d_in_currency"),
        rsi, coin_data.get("current_price"), coin_data.get("market_cap"), coin_data.get("total_volume"),
        "Medium", horizon_days, headlines[:5], forecast_note
    )

    # 6. Table Build
    forecast_table = []
    last_date = hist_df.index[-1] if not hist_df.empty else datetime.now()
    for i in range(horizon_days):
        dt = last_date + pd.Timedelta(days=i+1)
        p_val = prophet_df["yhat"].iloc[-(horizon_days-i)] if not prophet_df.empty else None
        l_val = lstm_future[i] if i < len(lstm_future) else None
        e_val = ensemble_preds[i] if i < len(ensemble_preds) else None
        forecast_table.append({"Date": dt, "Prophet": p_val, "LSTM": l_val, "XGBoost": e_val})

    return {
        "coin_id": coin_id, "coin_symbol": coin_symbol,
        "price": coin_data.get("current_price"),
        "pct_24h": coin_data.get("price_change_percentage_24h"),
        "pct_7d": coin_data.get("price_change_percentage_7d_in_currency"),
        "market_cap": coin_data.get("market_cap"),
        "volume_24h": coin_data.get("total_volume"),
        "rsi": rsi,
        "agg_sent": agg_sent,
        "sentiment_pcts": sentiment_pcts,
        "sentiment_table": sentiment_table,
        "history": hist_df,
        "prophet_df": prophet_df,
        "lstm_preds": lstm_future,
        "ensemble_preds": ensemble_preds, # New key
        "forecast_table": forecast_table,
        "recommendation": rec,
        "articles": articles
    }

def make_pretty_output(result, comps, horizon):
    rec = result["recommendation"]
    lines = [
        f"# {result['coin_id'].upper()} Analysis ({horizon} Days)",
        f"**Price**: ${result['price']:,}",
        f"**Recommendation**: {rec.get('rating')} (Score: {rec.get('score')})",
        f"**Sentiment**: {result['agg_sent']:.3f} (Twitter-RoBERTa)",
        "",
        f"**XGBoost Forecast**: ${result['ensemble_preds'][-1]:.2f}" if result['ensemble_preds'] else "Forecast N/A",
        f"**Insight**: {rec.get('insight')}"
    ]
    return "\n".join(lines)

def build_single_response(msg, sid):
    # Intent parsing
    coin_id = "bitcoin" # simplistic fallback
    for c in DEFAULT_COINS:
        if c["id"] in msg.lower() or c["symbol"] in msg.lower(): coin_id = c["id"]
    
    horizon = 7
    if "30" in msg: horizon = 30
    
    save_conversation(sid, "user", msg)
    res = analyze_and_forecast(coin_id, coin_id, horizon, msg)
    
    # Defensive check
    if "error" in res:
        return f"Error: {res['error']}", {}, "", None, res

    comps, _ = explain_trace_components(res["agg_sent"], res["pct_24h"], res["pct_7d"], res["rsi"])
    pretty = make_pretty_output(res, comps, horizon)
    
    # Headlines
    heads = "\n".join([f"- {r['text']} ({r['label']})" for _, r in res["sentiment_table"].head(5).iterrows()])
    
    # Chart
    chart = plot_history_forecasts_to_file(res["history"], res["prophet_df"], res["lstm_preds"], res["ensemble_preds"], coin_id)
    
    save_conversation(sid, "assistant", pretty, {"chart": chart})
    return pretty, comps, heads, chart, res

def render_pretty_summary(res, horizon_days):
    if not res or "error" in res:
        st.error(res.get("error", "Unknown error"))
        return
        
    coin_id = res.get("coin_id", "Unknown")
    
    st.subheader(f"{coin_id.title()} Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${res['price']:,}")
    c2.metric("Sentiment", f"{res['agg_sent']:.2f}")
    c3.metric("RSI", f"{res['rsi']:.1f}")
    
    st.info(f"**Recommendation**: {res['recommendation'].get('rating')} \n\n {res['recommendation'].get('insight')}")

    t1, t2, t3 = st.tabs(["Chart", "Data", "Headlines"])
    with t1:
        # Altair chart logic could go here, for now using static image if generated, or basic line
        if res.get("history") is not None:
            st.line_chart(res["history"]["price"])
            
    with t2:
        if res.get("forecast_table"):
            st.dataframe(pd.DataFrame(res["forecast_table"]))
            
    with t3:
        if not res["sentiment_table"].empty:
            st.dataframe(res["sentiment_table"][["text", "label", "score"]])

# ---------------------------
# UI (Preserved Structure)
# ---------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.last_outputs = {}

st.title("Crypto Analyst (XGBoost + RoBERTa)")

with st.container():
    st.markdown("<div class='card input-card'>", unsafe_allow_html=True)
    user_message = st.text_input("Message", placeholder="e.g. BTC forecast")
    send_clicked = st.button("Send", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if send_clicked and user_message:
    p_text, ex, h_text, ch, res_obj = build_single_response(user_message, st.session_state.session_id)
    st.session_state.last_outputs = {
        "result_for_ui": res_obj,
        "horizon": 7 # simplified
    }

if st.session_state.last_outputs.get("result_for_ui"):
    render_pretty_summary(st.session_state.last_outputs["result_for_ui"], 7)
