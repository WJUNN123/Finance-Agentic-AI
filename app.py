# app.py
import os
import streamlit as st

from agents import run_agent
import tools as tool_impl

st.set_page_config(page_title="Agentic Gemini (Router + Tools)", layout="wide")
st.title("Agentic AI (Gemini Router + Summary) — Streamlit Demo")
st.caption("Informational only. Not financial advice. No automated trading actions.")

# API key from Streamlit Secrets ONLY
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# Sidebar settings (NO model selector now)
st.sidebar.header("Settings")
max_steps = st.sidebar.slider("Max tool steps", 2, 10, 6)
show_debug = st.sidebar.checkbox("Show debug details", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Example prompts:")
st.sidebar.code("Summarize BTCUSDT on 1h timeframe. Include indicators.")
st.sidebar.code("Give me ETHUSDT 4h snapshot with risk label.")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_input = st.chat_input("Ask about BTC/ETH market conditions (e.g., 'BTCUSDT 1h summary')")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking + calling tools..."):
            answer, new_history = run_agent(
                user_message=user_input,
                chat_history=st.session_state.history,
                max_steps=max_steps,
            )
            st.markdown(answer)

    st.session_state.history = new_history
    st.session_state.messages.append({"role": "assistant", "content": answer})

    if show_debug:
        st.markdown("### Debug (latest agent history items)")
        try:
            st.json([c.model_dump() for c in st.session_state.history[-6:]])
        except Exception:
            st.write(st.session_state.history[-6:])

# Optional manual chart (CoinGecko)
st.markdown("---")
st.subheader("Optional: Quick chart (manual fetch)")

col1, col2, col3 = st.columns(3)
symbol = col1.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)
interval = col2.selectbox("Interval", ["15m", "30m", "1h", "4h", "1d"], index=2)
limit = col3.slider("Candles", 50, 500, 200, 50)

if st.button("Fetch & plot"):
    data = tool_impl.get_klines(symbol=symbol, interval=interval, limit=limit)

    if "error" in data:
        st.error(data["error"])
        st.json(data)
    else:
        import pandas as pd
        df = pd.DataFrame(data.get("candles", []))
        if df.empty:
            st.error("No candles returned.")
            st.json(data)
        else:
            df["open_time"] = pd.to_datetime(df["open_time"])
            df = df.sort_values("open_time")
            st.line_chart(df.set_index("open_time")["close"])
            st.json(tool_impl.compute_indicators(symbol=symbol, interval=interval, limit=limit))
