# agents.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from google import genai
from google.genai import types

import tools as tool_impl

ToolFn = Callable[..., Dict[str, Any]]

SYSTEM_PROMPT = """You are a router-style agent inside a Streamlit app.

You have tools:
- compute_indicators(symbol, interval, limit): returns latest price + SMA/RSI/volatility and metadata
- risk_label(volatility_50, rsi_14): returns a simple risk tag
- get_klines exists but prefer compute_indicators first.

When the user asks for a market snapshot / indicators / risk:
1) Call compute_indicators using the user's symbol and timeframe. Use limit=200 if not given.
2) If compute_indicators returns error, explain briefly and suggest retry.
3) If indicators succeeded, call risk_label using volatility_50 and rsi_14.
4) Write the final response in simple words.

Safety rules:
- Not financial advice.
- Do NOT give "buy/sell" commands.
- Use neutral labels such as "watch", "caution", "positive bias".

Output format:
1) Market snapshot (latest price, trend_hint, volatility)
2) Indicators (SMA20 vs SMA50, RSI14) explained simply
3) Risk label
4) Short conclusion (watch/caution/positive bias)
"""


def build_tools() -> Tuple[List[types.Tool], Dict[str, ToolFn]]:
    compute_indicators_decl = {
        "name": "compute_indicators",
        "description": "Compute latest price + SMA(20), SMA(50), RSI(14), volatility using fetched candles (has fallback).",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading pair like BTCUSDT, ETHUSDT."},
                "interval": {"type": "string", "description": "Interval: 15m, 1h, 4h, 1d."},
                "limit": {"type": "integer", "description": "Candles to use (recommended 200)."},
            },
            "required": ["symbol"],
        },
    }

    risk_label_decl = {
        "name": "risk_label",
        "description": "Create a simple risk label from volatility and RSI (informational only).",
        "parameters": {
            "type": "object",
            "properties": {
                "volatility_50": {"type": "number", "description": "Std dev of returns over last 50 candles."},
                "rsi_14": {"type": "number", "description": "RSI(14) value."},
            },
            "required": ["volatility_50", "rsi_14"],
        },
    }

    # Keep get_klines available (optional), but the prompt tells model to prefer compute_indicators
    get_klines_decl = {
        "name": "get_klines",
        "description": "Fetch recent OHLCV candles (Binance with fallback).",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading pair like BTCUSDT, ETHUSDT."},
                "interval": {"type": "string", "description": "Interval: 15m, 1h, 4h, 1d."},
                "limit": {"type": "integer", "description": "Number of candles."},
            },
            "required": ["symbol"],
        },
    }

    toolset = types.Tool(
        function_declarations=[compute_indicators_decl, risk_label_decl, get_klines_decl]
    )

    tool_map: Dict[str, ToolFn] = {
        "compute_indicators": tool_impl.compute_indicators,
        "risk_label": tool_impl.risk_label,
        "get_klines": tool_impl.get_klines,
    }

    return [toolset], tool_map


def run_agent(
    user_message: str,
    chat_history: List[types.Content],
    model: str = "gemini-2.5-flash",
    max_steps: int = 6,
) -> Tuple[str, List[types.Content]]:
    """
    Tool-loop agent:
      - Model may call tool(s)
      - App executes tool(s) and returns results
      - Model writes the final answer
    """
    client = genai.Client()
    tools, tool_map = build_tools()

    contents: List[types.Content] = []
    if chat_history:
        contents.extend(chat_history)

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=f"{SYSTEM_PROMPT}\n\nUser request: {user_message}")]
        )
    )

    config = types.GenerateContentConfig(
        tools=tools,
        temperature=0.2,
    )

    for _ in range(max_steps):
        resp = client.models.generate_content(model=model, contents=contents, config=config)

        model_parts = resp.candidates[0].content.parts
        contents.append(types.Content(role="model", parts=model_parts))

        fn_call = None
        for p in model_parts:
            if p.function_call:
                fn_call = p.function_call
                break

        if fn_call:
            fn_name = fn_call.name
            fn_args = dict(fn_call.args or {})
            tool_fn = tool_map.get(fn_name)

            if tool_fn is None:
                tool_result = {"error": f"Tool '{fn_name}' not implemented."}
            else:
                try:
                    tool_result = tool_fn(**fn_args)
                except Exception as e:
                    tool_result = {"error": f"{type(e).__name__}: {str(e)}", "args": fn_args}

            tool_part = types.Part.from_function_response(name=fn_name, response=tool_result)
            contents.append(types.Content(role="user", parts=[tool_part]))
            continue

        return (resp.text or ""), contents

    return "I reached the tool-step limit. Please try a shorter request.", contents
