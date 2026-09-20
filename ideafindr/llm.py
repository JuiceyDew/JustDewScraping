"""Single LLM entry point -> Ollama Cloud via its OpenAI-compatible API.

Everything that talks to a model goes through here, so switching provider or
model is a config change, not a code change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
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


# A small in-process response cache. Labelling the same corpus twice -- a
# re-analyse, or two runs over overlapping documents -- otherwise pays for the
# same completion again. Keyed on the exact request, so it can only ever return a
# byte-identical answer; disable with IDEAFINDR_LLM_CACHE=0.
_CACHE: "OrderedDict[str, str]" = OrderedDict()
_CACHE_MAX = 256
_CACHE_ENABLED = os.environ.get("IDEAFINDR_LLM_CACHE", "1") != "0"

# Running totals, logged at the end of a pipeline so the cost is visible.
_USAGE = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}


def usage_summary() -> dict[str, int]:
    return dict(_USAGE)


def reset_usage() -> None:
    for k in _USAGE:
        _USAGE[k] = 0


def _cache_key(model: str, system: str, prompt: str, temperature: float, max_tokens: int) -> str:
    h = hashlib.sha256()
    for part in (model, system, prompt, f"{temperature}", f"{max_tokens}"):
        h.update(part.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


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
    max_tokens: int | None = None,
) -> str:
    name = model or settings.fast_model
    budget = max_tokens or settings.llm_max_tokens

    key = _cache_key(name, system, prompt, temperature, budget)
    if _CACHE_ENABLED and key in _CACHE:
        _CACHE.move_to_end(key)
        _USAGE["cached"] += 1
        return _CACHE[key]

    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    r = client().chat.completions.create(
        model=name,
        messages=msgs,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=budget,
    )
    choice = r.choices[0]
    content = (choice.message.content or "").strip()

    _USAGE["calls"] += 1
    if (u := getattr(r, "usage", None)) is not None:
        _USAGE["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        _USAGE["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)

    if not content:
        # Reasoning models on this endpoint (kimi-k3, glm-5.3) emit a `reasoning`
        # field first and can exhaust max_tokens before writing any content. A
        # silent empty string here turns into a confusing failure much further
        # downstream, so name the cause at the source.
        reasoning = getattr(choice.message, "reasoning", None) or ""
        raise EmptyCompletion(
            f"{name} returned no content (finish_reason={choice.finish_reason}, "
            f"{len(reasoning)} chars of reasoning). It spent the token budget "
            f"reasoning. Raise llm_max_tokens (currently {budget}) or use a model "
            f"that does not do this -- gpt-oss:120b is the tested default."
        )

    if _CACHE_ENABLED:
        _CACHE[key] = content
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
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
    max_tokens: int | None = None,
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
        raw = chat(prompt, system=sys_msg, model=model,
                   temperature=0.1 + 0.2 * i, max_tokens=max_tokens)
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


class _BatchItem(BaseModel):
    """One labelled item in a batched labelling call."""

    index: int
    label: str = ""
    description: str = ""
    stance: str = "neutral"
    coherent: bool = True


class _Batch(BaseModel):
    items: list[_BatchItem]


def chat_json_batch(
    blocks: list[str],
    system: str,
    instructions: str,
    model: str | None = None,
    attempts: int = 3,
    per_block_max_tokens: int = 160,
    max_blocks_per_call: int = 12,
) -> list[_BatchItem | None]:
    """Label many clusters in a few calls instead of one call per cluster.

    Labelling a 19-theme run was 19 sequential requests, each paying full prompt
    overhead and latency. Batching sends several clusters together and asks for a
    JSON array back -- the single biggest reduction in Ollama usage in the
    pipeline.

    Blocks are chunked at `max_blocks_per_call` so the prompt stays bounded: a
    28-theme corpus costs three calls, not one oversized request that models
    truncate. Returns results positionally aligned to `blocks`; a block the model
    omitted is None, so the caller can fall back per item rather than losing the
    whole batch.
    """
    if not blocks:
        return []

    out: list[_BatchItem | None] = [None] * len(blocks)
    for start in range(0, len(blocks), max_blocks_per_call):
        chunk = blocks[start : start + max_blocks_per_call]
        prompt = _batch_prompt(chunk, instructions)
        budget = max(1024, per_block_max_tokens * len(chunk))
        # Clamp to the configured ceiling: a huge corpus must not ask for a giant
        # completion that a reasoning model will spend entirely on `reasoning`.
        budget = min(budget, max(settings.llm_max_tokens, 2048))
        try:
            result = chat_json(prompt, _Batch, system=system, model=model,
                               attempts=attempts, max_tokens=budget)
        except Exception as e:  # noqa: BLE001 - one bad chunk must not sink the rest
            log.warning("batch chunk %d-%d failed: %s", start, start + len(chunk), e)
            continue
        seen: set[int] = set()
        for item in result.items:
            if 0 <= item.index < len(chunk) and item.index not in seen:
                seen.add(item.index)
                out[start + item.index] = item
    return out


def _batch_prompt(blocks: list[str], instructions: str) -> str:
    numbered = "\n\n".join(f"### Cluster {i}\n{b}" for i, b in enumerate(blocks))
    return (
        f"{instructions}\n\n"
        f"Label EACH of the {len(blocks)} clusters below. Return JSON exactly as:\n"
        '{"items": [{"index": 0, "label": "...", "description": "...", '
        '"stance": "...", "coherent": true}, ...]}\n'
        "Include one item per cluster, using the cluster's number as `index`. "
        "Do not merge, reorder or omit clusters.\n\n"
        f"{numbered}"
    )


def available_models() -> list[str]:
    return sorted(m.id for m in client().models.list().data)
