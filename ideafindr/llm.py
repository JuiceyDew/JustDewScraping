"""Single LLM entry point -> Ollama Cloud via its OpenAI-compatible API.

Everything that talks to a model goes through here, so switching provider or
model is a config change, not a code change.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from ideafindr.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMNotConfigured(RuntimeError):
    pass


class EmptyCompletion(RuntimeError):
    """The model returned no content, having spent its budget on reasoning."""


def client() -> OpenAI:
    if not settings.ollama_api_key:
        raise LLMNotConfigured(
            "OLLAMA_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://ollama.com/settings/keys"
        )
    return OpenAI(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def chat(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    name = model or settings.fast_model
    r = client().chat.completions.create(
        model=name,
        messages=msgs,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = r.choices[0]
    content = (choice.message.content or "").strip()

    if not content:
        # Reasoning models on this endpoint (kimi-k3, glm-5.3) emit a `reasoning`
        # field first and can exhaust max_tokens before writing any content. A
        # silent empty string here turns into a confusing failure much further
        # downstream, so name the cause at the source.
        reasoning = getattr(choice.message, "reasoning", None) or ""
        raise EmptyCompletion(
            f"{name} returned no content (finish_reason={choice.finish_reason}, "
            f"{len(reasoning)} chars of reasoning). It spent the token budget "
            f"reasoning. Raise max_tokens (currently {max_tokens}) or use a model "
            f"that does not do this -- gpt-oss:120b is the tested default."
        )
    return content


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def _extract_json(text: str) -> str:
    """Models wrap JSON in prose or fences regardless of instructions."""
    if m := _FENCE.search(text):
        text = m.group(1)
    text = text.strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        return text
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def chat_json(
    prompt: str,
    schema: type[T],
    system: str = "",
    model: str | None = None,
    attempts: int = 3,
) -> T:
    """Chat, parse JSON, validate against a pydantic model, with repair retries.

    Ollama Cloud's OpenAI shim does not reliably support response_format, so this
    validates defensively rather than trusting structured-output support.
    """
    sys_msg = (
        f"{system}\n\nRespond with ONLY valid JSON matching this schema. "
        f"No prose, no markdown fences.\nSchema: {json.dumps(schema.model_json_schema())}"
    ).strip()

    last = ""
    for i in range(attempts):
        raw = chat(prompt, system=sys_msg, model=model, temperature=0.1 + 0.2 * i)
        last = raw
        try:
            return schema.model_validate_json(_extract_json(raw))
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            log.warning("chat_json attempt %d/%d failed: %s", i + 1, attempts, str(e)[:160])
            prompt = (
                f"{prompt}\n\nYour previous reply could not be parsed ({str(e)[:200]}). "
                "Return ONLY the JSON object."
            )
    raise ValueError(f"LLM did not return valid JSON after {attempts} attempts: {last[:400]}")


def available_models() -> list[str]:
    return sorted(m.id for m in client().models.list().data)
