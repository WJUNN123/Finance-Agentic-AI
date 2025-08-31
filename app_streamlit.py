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
    kwargs = {}
    if HF_TOKEN:
        kwargs["use_auth_token"] = True
    # Force pipeline to use PyTorch backend instead of TensorFlow/Keras
    FINBERT_PIPE = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        device=PIPELINE_DEVICE,
        framework="pt",  
        **kwargs
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
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    try:
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        summary = {}
        if not forecast.empty:
            future_rows = forecast.tail(days)
            last_hat = float(future_rows["yhat"].iloc[-1])
            last_upper = float(future_rows["yhat_upper"].iloc[-1])
            last_lower = float(future_rows["yhat_lower"].iloc[-1])
            summary = {"pred_last": last_hat, "upper": last_upper, "lower": last_lower}
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
    if price_series is None or len(price_series) < window + 10:
        return []

    series = price_series.astype(float).reset_index(drop=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm_model((window, 1))
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=LSTM_BATCH, callbacks=[es], verbose=0)

    preds_scaled = []
    last_window = scaled[-window:].tolist()
    for _ in range(horizon):
        x_in = np.array(last_window[-window:]).reshape((1, window, 1))
        yhat = model.predict(x_in, verbose=0)[0][0]
        preds_scaled.append(yhat)
        last_window.append(yhat)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten().tolist()
    return preds

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
            ensemble_vals = (np.array(lstm_preds) + prophet_vals) / 2.0
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

def recommend_and_insight(sentiment: float, pct_24h: float, pct_7d: float, rsi: float, risk: str, horizon_days: int) -> Dict:
    # Scale values to improve readability
    mom24 = scale(pct_24h, -15, 15)
    mom7 = scale(pct_7d, -40, 40)
    rsi_dev = 0.0
    if not (isinstance(rsi, float) and math.isnan(rsi)):
        if rsi >= 70: rsi_dev = -1.0
        elif rsi <= 30: rsi_dev = 1.0
        else: rsi_dev = (50 - rsi) / 20.0

    # Weights for sentiment, momentum, and RSI components
    w_sent, w_m24, w_m7, w_rsi = 0.45, 0.2, 0.2, 0.15
    risk = (risk or "Medium").lower()

    if risk == "low":
        w_m24 *= 0.8; w_m7 *= 0.8; w_rsi *= 1.2; w_sent *= 1.2
        threshold_buy, threshold_sell = 0.35, -0.25
    elif risk == "high":
        w_m24 *= 1.2; w_m7 *= 1.2; w_rsi *= 0.8; w_sent *= 0.8
        threshold_buy, threshold_sell = 0.25, -0.35
    else:
        threshold_buy, threshold_sell = 0.3, -0.3

    # Adjust thresholds for different horizon days
    if horizon_days <= 7:
        threshold_buy += 0.05; threshold_sell -= 0.05
    elif horizon_days >= 90:
        threshold_buy -= 0.05; threshold_sell += 0.05

    # Calculate overall score
    score = (w_sent * sentiment + w_m24 * mom24 + w_m7 * mom7 + w_rsi * rsi_dev)

    # Determine recommendation based on the score
    if score >= threshold_buy:
        rating = "BUY (speculative)" if risk == "high" and horizon_days <= 14 else "BUY"
    elif score <= threshold_sell:
        rating = "SELL / AVOID"
    else:
        rating = "HOLD / WAIT"

    # Build the insight
    pieces = []
    if sentiment >= 0.4: pieces.append("**Sentiment**: Strong positive news sentiment. 📈")
    elif sentiment >= 0.15: pieces.append("**Sentiment**: Moderately positive news sentiment. 📈")
    elif sentiment <= -0.4: pieces.append("**Sentiment**: Strong negative news sentiment. 📉")
    elif sentiment <= -0.15: pieces.append("**Sentiment**: Moderately negative news sentiment. 📉")
    else: pieces.append("**Sentiment**: Neutral/mixed news sentiment. ⚖️ This suggests market uncertainty, and it's best to wait for clearer signals before making significant moves.")

    if pct_24h is not None and not (isinstance(pct_24h, float) and math.isnan(pct_24h)):
        if pct_24h > 5: pieces.append("**24-hour Momentum**: Strong short-term upward momentum. 📈")
        elif pct_24h < -5: pieces.append("**24-hour Momentum**: Strong short-term downward momentum. 📉")
        else: pieces.append("**24-hour Momentum**: No strong short-term momentum. ⚖️ The lack of significant price movement means you might want to hold off on any new investments for now.")

    if pct_7d is not None and not (isinstance(pct_7d, float) and math.isnan(pct_7d)):
        if pct_7d > 10: pieces.append("**7-day Momentum**: 7-day trend is strongly positive. 📈")
        elif pct_7d < -10: pieces.append("**7-day Momentum**: 7-day trend is strongly negative. 📉")
        else: pieces.append("**7-day Momentum**: Mild/moderate trend. 🔄 Over the past week, prices have remained relatively stable. It’s a signal that the market is in a consolidation phase.")

    if not (isinstance(rsi, float) and math.isnan(rsi)):
        if rsi >= 70: pieces.append("**RSI (14)**: RSI is high → market may be overbought. 📉")
        elif rsi <= 30: pieces.append("**RSI (14)**: RSI is low → market may be oversold. 📈")
        else: pieces.append("**RSI (14)**: Neutral range. 🧠 The current market is neither overbought nor oversold, implying a balanced risk. A neutral RSI suggests it's neither a great time to buy nor to sell.")

    insight_text = "\n\n".join(pieces)
    return {"rating": rating, "score": float(score), "insight": insight_text}

# ---------------------------
# Short-term memory (SQLite)
# ---------------------------
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

# ---------------------------
# Long-term memory: embeddings + FAISS (if available)
# ---------------------------
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
            FAISS_META.extend(metas)
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

# ---------------------------
# Continual retraining helpers (skeleton)
# ---------------------------
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
    joblib.dump({"coin": coin_id, "saved_at": ts, "shape": X.shape}, path + ".meta.pkl")
    return True, path

# ---------------------------
# Planner stubs
# ---------------------------
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

# ---------------------------
# Explainability helpers
# ---------------------------
def explain_trace_components(agg_sent, pct_24h, pct_7d, rsi):
    mom24 = scale(pct_24h, -15, 15)
    mom7 = scale(pct_7d, -40, 40)

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

# ---------------------------
# Full analyze_coin (enhanced)
# ---------------------------
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

    rec = recommend_and_insight(agg_sent, pct_24h, pct_7d, rsi, risk, horizon_days)

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
            ensemble_val = float((prophet_val + lstm_val) / 2.0)
        elif prophet_val is not None:
            ensemble_val = float(prophet_val)
        elif lstm_val is not None:
            ensemble_val = float(lstm_val)
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
        "top_headlines": df_sent.head(3).to_dict(orient="records") if not df_sent.empty else []
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
        "recommendation": rec,
        "prophet_df": prophet_df,
        "prophet_summary": prophet_summary,
        "lstm_preds": lstm_preds,
        "forecast_table": forecast_table,
        "last_prediction": last_prediction,
        "explainability": explain_trace,
    }

# ---------------------------
# Intent parser & pretty output
# ---------------------------
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
    return {"coin_id": coin_id, "intent": intent, "horizon_days": horizon_days}


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
        comp_line("RSI",       "rsi_component",       "price momentum is improving",      "potential overbought/oversold", "balanced momentum"),
        comp_line("Sentiment", "sentiment_component", "news sentiment is mildly positive","news sentiment is mildly negative","mixed/neutral news flow"),
        comp_line("Mom24",     "mom24_component",     "24-hour momentum positive",        "flat 24-hour momentum",         "flat 24-hour momentum"),
        comp_line("Mom7",      "mom7_component",      "7-day momentum uptrend",           "7-day momentum almost flat",    "flat 7-day momentum"),
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

# ---------------------------
# Build response for chat (single pipeline)
# ---------------------------
def build_single_response(user_message: str, session_id: str):
    parsed = parse_user_message(user_message)
    coin_id = parsed["coin_id"]
    h = parsed["horizon_days"]
    coin_symbol = next((c["symbol"] for c in DEFAULT_COINS if c["id"]==coin_id), coin_id[:4])

    save_conversation(session_id, "user", user_message)

    result = analyze_coin(coin_id, coin_symbol, risk="Medium", horizon_days=h, forecast_days=h, model_choice="ensemble")
    if "error" in result:
        resp = f"Error: {result['error']}"
        save_conversation(session_id, "assistant", resp)
        return resp, {}, "", None

    pretty = make_pretty_output(result, h)

    ex = result.get("explainability", {})
    comps = ex.get("components", {})

    friendly_ex = []
    total_score = comps.get("total_score", None)
    if total_score is not None:
        exam_score = int(round(50 + 50 * total_score))
        exam_score = max(0, min(100, exam_score))
        if exam_score >= 80:
            mood_text = "very bullish 📈"
        elif exam_score >= 60:
            mood_text = "slightly bullish 📈"
        elif exam_score >= 40:
            mood_text = "neutral ⚖️"
        elif exam_score >= 20:
            mood_text = "slightly bearish 📉"
        else:
            mood_text = "very bearish 📉"
        friendly_ex.append(f"Score: {exam_score} → {mood_text}")

    def describe_component(label, value, meaning):
        if value > 0.02:
            trend = "slightly bullish"
        elif value < -0.02:
            trend = "slightly bearish"
        else:
            trend = "neutral"
        return f"{label}: {value:+.4f} → {trend} ({meaning})"

    if "rsi_component" in comps:
        friendly_ex.append(describe_component("RSI", comps["rsi_component"], "price momentum is improving" if comps["rsi_component"] > 0 else "potential overbought/oversold"))
    if "sentiment_component" in comps:
        friendly_ex.append(describe_component("Sentiment", comps["sentiment_component"], "news sentiment is mildly positive" if comps["sentiment_component"] > 0 else "news sentiment is mildly negative"))
    if "mom24_component" in comps:
        friendly_ex.append(describe_component("Mom24", comps["mom24_component"], "24-hour momentum positive" if comps["mom24_component"] > 0 else "24-hour momentum weak"))
    if "mom7_component" in comps:
        friendly_ex.append(describe_component("Mom7", comps["mom7_component"], "7-day momentum uptrend" if comps["mom7_component"] > 0 else "7-day momentum almost flat"))

    friendly_ex_text = "\n".join(friendly_ex)

    headlines_text = ""
    if not result.get("sentiment_table", pd.DataFrame()).empty:
        dfh = result["sentiment_table"]
        headlines_text = "\n".join([f"{r['text']} → {r['label']} ({r['score']:.2f})" for _, r in dfh.head(6).iterrows()])
    else:
        headlines_text = "\n".join([a.get("title","") for a in result.get("articles",[])[:6]])

    chart_path = plot_history_forecasts_to_file(result.get("history", pd.DataFrame()), result.get("prophet_df", pd.DataFrame()), result.get("lstm_preds", []), coin_id)

    save_conversation(session_id, "assistant", pretty, {"chart": chart_path, "explain_summary": friendly_ex_text})
    save_conversation(session_id, "assistant", json.dumps(ex), {"explain_full": True})

    return pretty, ex, headlines_text, chart_path

# ---------------------------
# STREAMLIT UI (replacing Gradio fully)
# ---------------------------
st.title("💬 Crypto Analyst (Streamlit)")
st.caption("Ask about BTC, ETH, SOL, etc. Single-send → summary, explainability, headlines, chart. Educational only — not financial advice.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_outputs" not in st.session_state:
    st.session_state.last_outputs = {"pretty": "", "ex": {}, "heads": "", "chart": None}

col_in1, col_in2 = st.columns([4,1])
with col_in1:
    user_message = st.text_input("Your message", placeholder="Type your question (e.g. 'Should I buy Bitcoin?')")
with col_in2:
    send_clicked = st.button("Send", use_container_width=True)

if send_clicked and user_message.strip():
    pretty_text, full_ex, headlines_text, chart_path = build_single_response(user_message, st.session_state.session_id)
    st.session_state.chat.append((user_message, pretty_text))
    st.session_state.last_outputs = {"pretty": pretty_text, "ex": full_ex or {}, "heads": headlines_text, "chart": chart_path}

# Chat transcript (simple)
for u, a in st.session_state.chat:
    with st.container(border=True):
        st.markdown(f"**You:** {u}")
        st.markdown(a)

# Tabs like in your Gradio app
summary_tab, explain_tab, headlines_tab, chart_tab = st.tabs(["📊 Summary", "🔎 Explainability", "📰 Headlines", "📈 Forecast Chart"])

with summary_tab:
    st.markdown(st.session_state.last_outputs["pretty"] or "_No summary yet._")

with explain_tab:
    # Friendly summary
    comps = (st.session_state.last_outputs["ex"] or {}).get("components", {})
    friendly = []
    total = comps.get("total_score", None)
    if total is not None:
        if total > 0.05:
            mood = "slightly bullish 📈"
        elif total < -0.05:
            mood = "slightly bearish 📉"
        else:
            mood = "neutral 😐"
        friendly.append(f"Model score: {total:.3f} → {mood}")
    for k in ["rsi_component", "sentiment_component", "mom24_component", "mom7_component"]:
        if k in comps:
            friendly.append(f"{k.replace('_',' ').capitalize()}: {comps[k]:+.4f}")
    st.text("\n".join(friendly) or "No explainability yet.")

    st.divider()
    st.subheader("Full Technical Trace")
    st.json(st.session_state.last_outputs["ex"] or {})

with headlines_tab:
    st.text_area("Top Headlines (with sentiment)", value=st.session_state.last_outputs["heads"], height=240)

with chart_tab:
    if st.session_state.last_outputs["chart"]:
        st.image(st.session_state.last_outputs["chart"], caption="Price & Forecast Chart")
    else:
        st.info("Run a query to see the chart.")
