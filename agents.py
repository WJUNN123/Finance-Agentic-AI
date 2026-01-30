# agents.py
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Any

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import tools as tool_impl

ToolFn = Callable[..., Dict[str, Any]]

SYSTEM_PROMPT = """You are a router-style agent inside a Streamlit app.

You have tools:
- compute_indicators(symbol, interval, limit): returns latest price + SMA/RSI/volatility and metadata
- risk_label(volatility_50, rsi_14): returns a simple risk tag
- get_klines(symbol, interval, limit): returns candles (CoinGecko based)

When the user asks for a market snapshot / indicators / risk:
1) Call compute_indicators using the user's symbol and timeframe. Use limit=200 if not given.
2) If compute_indicators returns an error, include the exact error message in your reply.
3) If indicators succeeded, call risk_label using volatility_50 and rsi_14.
4) Write the final response in simple words.

Safety rules:
- Not financial advice.
- Do NOT give "buy" / "sell" commands.
- Use neutral labels such as "watch", "caution", "positive bias".

Output format:
1) Market snapshot
2) Indicators (SMA + RSI) in simple terms
3) Risk label
4) Short conclusion
"""


def build_tools() -> Tuple[List[types.Tool], Dict[str, ToolFn]]:
    compute_indicators_decl = {
        "name": "compute_indicators",
        "description": "Compute latest price + SMA(20), SMA(50), RSI(14), volatility using candles.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading pair like BTCUSDT, ETHUSDT."},
                "interval": {"type": "string", "description": "Interval: 15m, 30m, 1h, 4h, 1d."},
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

    get_klines_decl = {
        "name": "get_klines",
        "description": "Fetch candles (CoinGecko market_chart + resampling).",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading pair like BTCUSDT, ETHUSDT."},
                "interval": {"type": "string", "description": "Interval: 15m, 30m, 1h, 4h, 1d."},
                "limit": {"type": "integer", "description": "Number of candles."},
            },
            "required": ["symbol"],
        },
    }

    toolset = types.Tool(function_declarations=[compute_indicators_decl, risk_label_decl, get_klines_decl])

    tool_map: Dict[str, ToolFn] = {
        "compute_indicators": tool_impl.compute_indicators,
        "risk_label": tool_impl.risk_label,
        "get_klines": tool_impl.get_klines,
    }

    return [toolset], tool_map


def run_agent(
    user_message: str,
    chat_history: List[types.Content],
    model: str = "gemini-1.5-flash",
    max_steps: int = 6,
) -> Tuple[str, List[types.Content]]:
    """
    Multi-step tool-loop agent:
      - Model calls tools
      - App executes tools and returns results
      - Model writes final answer
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
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.ClientError:
            return (
                "Gemini API ClientError. Common causes: wrong model name, invalid key, API not enabled, or quota limit. "
                "Check Streamlit Cloud logs for the exact HTTP status (401/403/404/429).",
                contents,
            )
        except Exception as e:
            return (f"Gemini API call failed: {type(e).__name__}: {e}", contents)

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
