"""LangChain integration example for halu.

Add hallucination detection to any LangChain chain. This is an *example*, not a
sub-library — copy the pieces you need. It uses only ``langchain_core``
primitives, so it works with ANY chat model (OpenAI, Anthropic, local, ...).

Three patterns, in increasing order of intervention:

  1. annotate  — run detection alongside the answer and pass the score through
                 (default; never blocks the user).
  2. gate      — raise / replace the answer when it's flagged at/above a
                 threshold (stop bad output from reaching the user).
  3. retry     — regenerate with hallucination-informed feedback until clean
                 (recover automatically).

Install::

    pip install halu langchain-core
    export HALU_API_KEY=sk_...          # from https://detector.komplexai.io/account/keys

Run::

    python examples/langchain_integration.py

The demo builds a tiny fake "model" so it runs without an LLM provider key.
Swap ``fake_model`` for your real chain (e.g. ``prompt | ChatOpenAI() | StrOutputParser()``).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import halu

try:
    from langchain_core.runnables import Runnable, RunnableLambda
except ImportError as exc:  # pragma: no cover - example guard
    raise SystemExit(
        "This example needs langchain-core. Install it with:\n"
        "    pip install langchain-core\n"
    ) from exc


# ---------------------------------------------------------------------------
# Helper: normalize a chain's output to a plain string.
# LangChain chat models return an AIMessage; StrOutputParser returns a str.
# ---------------------------------------------------------------------------
def _to_text(model_output: Any) -> str:
    if isinstance(model_output, str):
        return model_output
    content = getattr(model_output, "content", None)
    if isinstance(content, str):
        return content
    return str(model_output)


# ---------------------------------------------------------------------------
# Pattern 1 — annotate: attach the detection result, never block.
# Returns a dict so downstream steps (or your UI) can decide what to do.
# ---------------------------------------------------------------------------
def halu_annotate(prompt: Optional[str] = None) -> Runnable:
    """A Runnable that scores its input text and returns an annotated dict.

    Output shape::

        {"answer": str, "p_hallucination": float, "flag": bool, "top_regime": str}
    """

    def _run(model_output: Any) -> Dict[str, Any]:
        answer = _to_text(model_output)
        result = halu.detect(answer, prompt=prompt)
        return {
            "answer": answer,
            "p_hallucination": result.p_hallucination,
            "flag": result.flag,
            "top_regime": result.top_regime,
        }

    return RunnableLambda(_run)


# ---------------------------------------------------------------------------
# Pattern 2 — gate: raise when flagged so a caller can catch + substitute.
# ---------------------------------------------------------------------------
def halu_gate(threshold: float = 0.5, prompt: Optional[str] = None) -> Runnable:
    """A Runnable that passes the answer through, or raises when flagged.

    Raises :class:`halu.HaluHallucinationFlagged` (``.detection_result`` attached)
    when ``p_hallucination >= threshold``. Catch it to show a safe fallback.
    """

    def _run(model_output: Any) -> str:
        answer = _to_text(model_output)
        halu.detect_or_raise(answer, threshold=threshold, prompt=prompt)
        return answer

    return RunnableLambda(_run)


# ---------------------------------------------------------------------------
# Pattern 3 — retry: regenerate with feedback until clean.
# Wraps a *callable* that produces an answer (so we can call it repeatedly).
# ---------------------------------------------------------------------------
def generate_clean(
    ask: Callable[..., Any],
    prompt: str,
    *,
    max_retries: int = 2,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Call ``ask`` up to ``max_retries+1`` times, retrying while flagged.

    ``ask`` is any callable ``ask(prompt, feedback=None) -> answer`` — for a
    LangChain chain, wrap it as ``lambda p, feedback=None: chain.invoke(...)``.
    Retries include a corrective hint derived from the detected regime.
    """
    from halu import HaluRegenerationExhausted

    def _ask_text(p: str, feedback: Optional[str] = None) -> str:
        return _to_text(ask(p, feedback=feedback))

    try:
        answer, result, _history = halu.regenerate_until_clean(
            _ask_text,
            prompt=prompt,
            max_retries=max_retries,
            acceptance_threshold=threshold,
            feedback_detail="regime",
        )
        return {"answer": answer, "p_hallucination": result.p_hallucination, "clean": True}
    except HaluRegenerationExhausted as exc:
        best_answer, best_result = min(exc.history, key=lambda pair: pair[1].p_hallucination)
        return {"answer": best_answer, "p_hallucination": best_result.p_hallucination, "clean": False}


# ---------------------------------------------------------------------------
# Demo — runs with a fake model so no LLM provider key is required.
# ---------------------------------------------------------------------------
def _demo() -> None:
    if not os.environ.get("HALU_API_KEY"):
        print(
            "Set HALU_API_KEY to run this demo:\n"
            "    export HALU_API_KEY=sk_...   (from https://detector.komplexai.io/account/keys)"
        )
        return

    # A fake chain that "answers" with a known fabrication so the demo is deterministic.
    fabrication = "The Eiffel Tower was built in 1789 by Napoleon Bonaparte."
    clean_fact = "Paris is the capital of France."
    fake_model = RunnableLambda(lambda _q: fabrication)

    print("── Pattern 1: annotate ──")
    annotated = (fake_model | halu_annotate()).invoke("Who built the Eiffel Tower?")
    print(annotated)

    print("\n── Pattern 2: gate ──")
    try:
        (fake_model | halu_gate(threshold=0.5)).invoke("Who built the Eiffel Tower?")
        print("passed the gate")
    except halu.HaluHallucinationFlagged as e:
        r = e.detection_result
        print(f"BLOCKED: p={r.p_hallucination:.2f} regime={r.top_regime} → show a safe fallback instead")

    print("\n── Pattern 3: retry ──")
    # First attempt fabricates, second attempt returns a clean fact.
    attempts = [fabrication, clean_fact]

    def ask(_prompt: str, feedback: Optional[str] = None) -> str:
        return attempts.pop(0) if attempts else clean_fact

    print(generate_clean(ask, prompt="Where is Paris?", max_retries=2, threshold=0.5))


if __name__ == "__main__":
    _demo()
