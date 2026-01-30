# agents.py
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Any

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import tools as tool_impl

ToolFn = Callable[..., Dict[str, Any]]

# 🔒 LOCKED MODEL (NO 2.5)
MODEL_NAME = "gemini-1.5-flash"

SYSTEM_PROMPT = """You are a router-style agent inside a Streamlit app.

You have tools:
- compute_indicators(symbol, interval, limit)
- risk_label(volatility_50, rsi_14)
- get_klines(symbol, interval, limit)

Workflow:
1) Call compute_indicators first.
2) If it returns error, include the exact error text.
3) If successful, call risk_label.
4) Write a simple summary.

Rules:
- Not financial advice.
- No buy/sell commands.
- Use neutral terms like watch, caution, positive bias.
"""


def build_tools() -> Tuple[List[types.Tool], Dict[str, ToolFn]]:
    compute_indicators_decl = {
        "name": "compute_indicators",
        "description": "Compute SMA, RSI, volatility from candles.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "interval": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["symbol"],
        },
    }

    risk_label_decl = {
        "name": "risk_label",
        "description": "Create simple risk label.",
        "parameters": {
            "type": "object",
            "properties": {
                "volatility_50": {"type": "number"},
                "rsi_14": {"type": "number"},
            },
            "required": ["volatility_50", "rsi_14"],
        },
    }

    get_klines_decl = {
        "name": "get_klines",
        "description": "Fetch candles from CoinGecko.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "interval": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["symbol"],
        },
    }

    toolset = types.Tool(
        function_declarations=[
            compute_indicators_decl,
            risk_label_decl,
            get_klines_decl,
        ]
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
    max_steps: int = 6,
) -> Tuple[str, List[types.Content]]:

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
            resp = client.models.generate_content(
                model=MODEL_NAME,   # 🔒 HARDLOCKED
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as e:
            return (
                f"Gemini API ClientError. Most common causes:\n"
                f"- API not enabled\n"
                f"- Billing not enabled\n"
                f"- Invalid key\n"
                f"- Quota exceeded\n\n"
                f"Check Streamlit Cloud logs for HTTP status.",
                contents,
            )
        except Exception as e:
            return (f"Gemini API failed: {type(e).__name__}: {e}", contents)

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

            try:
                tool_result = tool_fn(**fn_args)
            except Exception as e:
                tool_result = {"error": f"{type(e).__name__}: {e}"}

            tool_part = types.Part.from_function_response(
                name=fn_name,
                response=tool_result,
            )

            contents.append(types.Content(role="user", parts=[tool_part]))
            continue

        return (resp.text or ""), contents

    return "Tool step limit reached.", contents
