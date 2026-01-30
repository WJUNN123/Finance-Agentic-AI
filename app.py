# app.py
import os
import streamlit as st
from google.genai import types

from agents import run_agent
import tools as tool_impl


st.set_page_config(page_title="Agentic Gemini (Router + Tools)", layout="wide")
st.title("Agentic AI (Gemini Router + Summary) — Streamlit Demo")
st.caption("Informational only. Not financial advice. No automated trading actions.")


# --- API key from Streamlit Secrets ONLY ---
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()


# Sidebar settings (safe)
st.sidebar.header("Settings")
model_name = st.sidebar.text_input("Model", value="gemini-2.5-flash")
max_steps = st.sidebar.slider("Max tool steps", 2, 10, 6)

st.sidebar.markdown("---")
st.sidebar.write("Example prompts:")
st.sidebar.code("Summarize BTCUSDT on 1h timeframe. Include indicators.")
st.sidebar.code("Give me ETHUSDT 4h snapshot with risk label.")


# Session state
if "history" not in st.session_state:
    st.session_state.history = []  # list[types.Content]
if "messages" not in st.session_state:
    st.session_state.messages = []  # for chat UI rendering


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
                model=model_name,
                max_steps=max_steps,
            )
            st.markdown(answer)

            st.session_state.history = new_history
            st.session_state.messages.append({"role": "assistant", "content": answer})


# Optional manual chart
st.markdown("---")
st.subheader("Opt
