# agents.py (replace run_agent with this)
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import tools as tool_impl

ToolFn = Callable[..., Dict[str, Any]]

# keep your SYSTEM_PROMPT + build_tools as before


def run_agent(
    user_message: str,
    chat_history: List[types.Content],
    model: str = "gemini-1.5-flash",   # <-- safer default
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
            resp = client.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.ClientError as e:
            # Streamlit redacts by default; we return a safe message
            return (
                "Gemini API error (ClientError). This usually means: wrong model name, API not enabled, "
                "invalid key, or quota/billing issue. Please check Streamlit Cloud logs for the exact "
                "HTTP status (401/403/404/429) and message.",
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
