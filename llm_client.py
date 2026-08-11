"""
Thin wrapper around Ollama so agent code doesn't repeat boilerplate.

Requires Ollama running locally: https://ollama.com
    ollama pull qwen2.5:7b-instruct     (or qwen2.5:14b-instruct if RAM allows)
"""

import json
import time
import ollama

DEFAULT_MODEL = "qwen2.5:7b-instruct"


def call_llm(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL,
             json_mode: bool = True, temperature: float = 0.3, max_tokens: int = 500,
             label: str = "") -> str:
    """Single call to local Ollama model. Returns raw text (or raw JSON string).
    max_tokens caps generation length -- structured verdicts don't need long
    outputs, and uncapped generation is a major source of slowness on CPU.
    label is just for the progress print so long runs aren't a silent black box."""
    if label:
        print(f"    -> calling model: {label}...", flush=True)
    t0 = time.time()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json" if json_mode else None,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    if label:
        print(f"    <- {label} done in {time.time() - t0:.1f}s", flush=True)
    return response["message"]["content"]


def call_llm_structured(system_prompt: str, user_prompt: str, schema_cls,
                         model: str = DEFAULT_MODEL, temperature: float = 0.3,
                         max_retries: int = 1, max_tokens: int = 500, label: str = ""):
    """
    Calls the model in JSON mode and parses into a pydantic model.
    Retries once on parse failure with an explicit correction nudge --
    small local models occasionally emit near-valid JSON. max_retries
    defaults to 1 (not 2) since each retry is a full extra model call.
    """
    prompt = user_prompt
    last_error = None
    for attempt in range(max_retries + 1):
        attempt_label = f"{label} (retry {attempt})" if attempt > 0 else label
        raw = call_llm(system_prompt, prompt, model=model, json_mode=True,
                        temperature=temperature, max_tokens=max_tokens, label=attempt_label)
        try:
            data = json.loads(raw)
            return schema_cls(**data)
        except Exception as e:
            last_error = e
            prompt = (
                user_prompt
                + f"\n\nYour previous response failed to parse. Error: {e}. "
                f"Follow EXACTLY the JSON schema described in the system prompt above -- "
                f"the field names, types, and nothing extra. All confidence-style fields "
                f"must be a plain number between 0.0 and 1.0, never a word like 'high' or "
                f"'moderate'. Return ONLY valid JSON, no markdown fences, no commentary."
            )
    raise ValueError(f"Failed to get valid structured output after {max_retries + 1} attempts: {last_error}")