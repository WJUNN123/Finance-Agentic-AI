# agents.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from google import genai
from google.genai import types

import tools as tool_impl

ToolFn = Callable[..., Dict[str, Any]]


SYSTEM_PROMPT = """You are a router-style agent inside a Streamlit app.

You can call tools to fetch market data and compute indicators.
You MUST call tools if the user asks for market conditions, prices, indicators, or risk.

Safety / constraints:
- You are NOT a financial advisor.
- Do NOT give "buy" / "sell" commands.
- Use neutral labels such as: "watch", "caution", "positive bias".
- Always explain reasons briefly.

Output format:
1) Market snapshot (price, trend, volatility)
2) Indicators (SMA, RSI) in simple terms
3) Risk label
4) Short conclusion (watch/caution/positive bias)
"""


def build_tools() -> Tuple[List[types.Tool], Dict[str, ToolFn]]:
    get_klines_decl = {
        "name": "get_klines",
        "description": "Fetch recent OHLCV candles for a symbol from Binance.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading pair like BTCUSDT, ETHUSDT."},
                "interval": {"type": "string", "description": "Interval: 1m, 5m, 15m, 1h, 4h, 1d."},
                "limit": {"type": "integer", "description": "Number of candles (e.g., 200)."},
            },
            "required": ["symbol"],
        },
    }

    compute_indicators_decl = {
        "name": "compute_indicators",
        "description": "Compute SMA(20), SMA(50), RSI(14), volatility using candles.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading pair like BTCUSDT, ETHUSDT."},
                "interval": {"type": "string", "description": "Interval: 1h, 4h, 1d."},
                "limit": {"type": "integer", "description": "Candles to use (e.g., 200)."},
            },
            "required": ["symbol"],
        },
    }

    risk_label_decl = {
        "name": "risk_label",
        "description": "Create a simple risk label from volatility and RSI (informational).",
        "parameters": {
            "type": "object",
            "properties": {
                "volatility_50": {"type": "number", "description": "Std dev of returns over last 50 candles."},
                "rsi_14": {"type": "number", "description": "RSI(14) value."},
            },
            "required": ["volatility_50", "rsi_14"],
        },
    }

    toolset = types.Tool(function_declarations=[get_klines_decl, compute_indicators_decl, risk_label_decl])

    tool_map: Dict[str, ToolFn] = {
        "get_klines": tool_impl.get_klines,
        "compute_indicators": tool_impl.compute_indicators,
        "risk_label": tool_impl.risk_label,
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

    # Add user content with system prompt (simple approach that works well in apps)
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

        # Save model output to history
        model_parts = resp.candidates[0].content.parts
        contents.append(types.Content(role="model", parts=model_parts))

        # Look for function call
        fn_call = None
        for p in model_parts:
            if p.function_call:
                fn_call = p.function_call
                break

        # If tool call exists, run it and feed it back
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

        # Otherwise return normal text
        return (resp.text or ""), contents

    return "I reached the tool-step limit. Please try a shorter request.", contents
