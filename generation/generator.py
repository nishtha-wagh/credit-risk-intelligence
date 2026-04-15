"""
generation/generator.py

Calls LLM with assembled context and returns a structured DecisionOutput.

Supports three providers via LLM_PROVIDER in .env:
  groq       → Groq API (free tier, fast, recommended)
  ollama     → local Ollama (fully free, no internet)
  openai     → OpenAI API (paid)
  anthropic  → Anthropic Claude API (paid)

Groq uses the OpenAI-compatible client — just set OPENAI_BASE_URL.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from generation.context_builder import XGBoostSignal, SHAPSignal, build_context

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT_PATH    = Path("prompts/system_prompt.txt")
DECISION_TEMPLATE_PATH = Path("prompts/decision_template.txt")


@dataclass
class DecisionOutput:
    borrower_id: str
    risk_tier: str
    confidence: float
    decision: str
    key_signals: list[str] = field(default_factory=list)
    reasoning: str = ""
    retrieval_score: float = 0.0
    sources_used: int = 0
    latency_ms: int = 0
    xgb_predicted_tier: str | None = None
    xgb_probability: float | None = None
    raw_response: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def generate_assessment(
    borrower_id: str,
    borrower_row: dict,
    retrieved_chunks: list,
    xgb_signal: XGBoostSignal | None = None,
    shap_signal: SHAPSignal | None = None,
) -> DecisionOutput:
    system_prompt = _load_prompt(SYSTEM_PROMPT_PATH)
    template      = _load_prompt(DECISION_TEMPLATE_PATH)

    context_block = build_context(borrower_row, retrieved_chunks, xgb_signal, shap_signal)
    user_message  = template.replace("{{CONTEXT}}", context_block).replace(
        "{{BORROWER_ID}}", borrower_id
    )

    t0         = time.time()
    raw        = _call_llm(system_prompt, user_message)
    latency_ms = int((time.time() - t0) * 1000)

    parsed = _parse_output(raw)
    avg_retrieval_score = (
        sum(c.similarity_score for c in retrieved_chunks) / len(retrieved_chunks)
        if retrieved_chunks else 0.0
    )

    return DecisionOutput(
        borrower_id=borrower_id,
        risk_tier=parsed.get("risk_tier", "UNKNOWN"),
        confidence=float(parsed.get("confidence", 0.0)),
        decision=parsed.get("decision", ""),
        key_signals=parsed.get("key_signals", []),
        reasoning=parsed.get("reasoning", ""),
        retrieval_score=round(avg_retrieval_score, 4),
        sources_used=len(retrieved_chunks),
        latency_ms=latency_ms,
        xgb_predicted_tier=xgb_signal.predicted_tier if xgb_signal else None,
        xgb_probability=xgb_signal.probability if xgb_signal else None,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# LLM routing
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_message: str) -> str:
    if LLM_PROVIDER == "ollama":
        return _call_ollama(system_prompt, user_message)
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, user_message)
    # groq + openai both use the OpenAI client
    # Groq: set OPENAI_BASE_URL=https://api.groq.com/openai/v1 + GROQ_API_KEY as OPENAI_API_KEY
    return _call_openai_compatible(system_prompt, user_message)


def _call_openai_compatible(system_prompt: str, user_message: str) -> str:
    """Works for OpenAI and Groq (OpenAI-compatible)."""
    from openai import OpenAI
    client = OpenAI()   # reads OPENAI_API_KEY + OPENAI_BASE_URL from env

    # Groq doesn't support response_format=json_object on all models
    # so we instruct via the system prompt and parse defensively
    kwargs = dict(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,
    )
    if LLM_PROVIDER == "openai":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_ollama(system_prompt: str, user_message: str) -> str:
    import httpx
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model":  LLM_MODEL,
        "prompt": f"{system_prompt}\n\n{user_message}",
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = httpx.post(f"{ollama_url}/api/generate", json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json().get("response", "")


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text().strip()
    if "system" in path.name:
        return (
            "You are a credit risk analyst AI. "
            "Analyse the provided borrower data and case notes. "
            "Respond ONLY with a valid JSON object — no preamble, no markdown fences."
        )
    return (
        "Borrower ID: {{BORROWER_ID}}\n\n"
        "{{CONTEXT}}\n\n"
        "Produce a credit risk assessment as a JSON object with keys: "
        "risk_tier (LOW|MEDIUM|HIGH|CRITICAL), confidence (0.0-1.0), decision (string), "
        "key_signals (list of strings), reasoning (string)."
    )


def _parse_output(raw: str) -> dict:
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "risk_tier":  "UNKNOWN",
            "confidence": 0.0,
            "decision":   "Parse error — see raw_response",
            "key_signals": [],
            "reasoning":  raw[:500],
        }
