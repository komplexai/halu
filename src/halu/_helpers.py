"""Three helper functions covering the most common usage patterns.

- :func:`detect_or_raise` — gate output (raise when flagged)
- :func:`detect_or_warn`  — annotate output (log a warning when flagged)
- :func:`regenerate_until_clean` — recover via LLM-informed retry

Each helper is a thin wrapper around :func:`halu.detect`. For anything not
covered here, call :func:`halu.detect` directly and build your own pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

from ._client import detect
from ._errors import HaluHallucinationFlagged, HaluRegenerationExhausted
from ._types import DetectResult

_DEFAULT_LOGGER = logging.getLogger("halu")

_REGIME_GUIDANCE = {
    "FABRICATED":     "avoid inventing facts that cannot be verified",
    "NEAR_FALSE":     "be precise about details (numbers, dates, names)",
    "SELF_CONTR":     "ensure internal consistency",
    "UNDERSPECIFIED": "be concrete and specific in your answer",
    "CF_AUTH":        "do not cite sources you cannot verify",
    "FALSE_REFUSAL":  "answer directly rather than refusing",
}

_BINARY_FEEDBACK = (
    "Previous response was flagged as a likely hallucination. "
    "Please regenerate more carefully."
)


def detect_or_raise(
    response: str,
    prompt: Optional[str] = None,
    threshold: float = 0.5,
    raise_on_flag: bool = True,
    **detect_kwargs: Any,
) -> DetectResult:
    """Run :func:`detect`. Raise :class:`HaluHallucinationFlagged` if flagged at threshold; return otherwise.

    Use when you want to stop bad output from being returned at all.

    Parameters
    ----------
    response : str
        The LLM-generated text to score.
    prompt : str | None
        Optional prompt the LLM saw.
    threshold : float
        User-side acceptance threshold applied to ``result.p_hallucination``.
        Defaults to ``0.5``. Lower = stricter (more raises); higher = looser.
    raise_on_flag : bool
        When ``False``, suppresses the raise and just returns the
        :class:`DetectResult`. Present for dev/test override.
    **detect_kwargs
        Forwarded to :func:`detect`: ``task``, ``api_key``, ``base_url``,
        ``timeout``.

    Returns
    -------
    DetectResult

    Raises
    ------
    HaluHallucinationFlagged
        ``result.p_hallucination >= threshold`` and ``raise_on_flag`` is ``True``.
    HaluError
        Any error from :func:`detect` is propagated unchanged.
    """
    result = detect(response, prompt=prompt, **detect_kwargs)
    if raise_on_flag and result.p_hallucination >= threshold:
        raise HaluHallucinationFlagged(
            message=(
                f"Response flagged as likely hallucination "
                f"(p={result.p_hallucination:.3f} >= threshold={threshold}, "
                f"top_regime={result.top_regime})."
            ),
            detection_result=result,
        )
    return result


def detect_or_warn(
    response: str,
    prompt: Optional[str] = None,
    threshold: float = 0.4,
    logger: Optional[logging.Logger] = None,
    **detect_kwargs: Any,
) -> DetectResult:
    """Run :func:`detect`. Log a warning if flagged at threshold; always return the :class:`DetectResult`.

    Use when you want monitoring/telemetry but no enforcement.

    Parameters
    ----------
    response : str
        The LLM-generated text to score.
    prompt : str | None
        Optional prompt the LLM saw.
    threshold : float
        User-side warning threshold applied to ``result.p_hallucination``.
        Defaults to ``0.4`` — lower than the gate default so warnings surface earlier.
    logger : logging.Logger | None
        Logger to emit the warning on. Defaults to ``logging.getLogger("halu")``.
    **detect_kwargs
        Forwarded to :func:`detect`: ``task``, ``api_key``, ``base_url``,
        ``timeout``.

    Returns
    -------
    DetectResult
        Always returned (never raises on flag — only on transport/auth errors).
    """
    result = detect(response, prompt=prompt, **detect_kwargs)
    if result.p_hallucination >= threshold:
        log = logger if logger is not None else _DEFAULT_LOGGER
        log.warning(
            "halu detected potential hallucination: p=%.3f >= threshold=%s, "
            "top_regime=%s, request_id=%s",
            result.p_hallucination,
            threshold,
            result.top_regime,
            result.request_id,
        )
    return result


def _build_feedback(result: DetectResult, feedback_detail: str) -> Optional[str]:
    if feedback_detail == "none":
        return None
    if feedback_detail == "binary":
        return _BINARY_FEEDBACK
    if feedback_detail == "regime":
        guidance = _REGIME_GUIDANCE.get(result.top_regime)
        if guidance is None:
            return _BINARY_FEEDBACK
        return (
            f"Previous response was flagged as {result.top_regime}. "
            f"Please regenerate, {guidance}."
        )
    raise ValueError(
        f"feedback_detail must be 'regime', 'binary', or 'none' (got {feedback_detail!r})."
    )


def regenerate_until_clean(
    generate_fn: Callable[..., str],
    prompt: str,
    max_retries: int = 2,
    acceptance_threshold: float = 0.5,
    on_exhausted: str = "raise",
    feedback_detail: str = "regime",
    **detect_kwargs: Any,
) -> Tuple[str, DetectResult, List[Tuple[str, DetectResult]]]:
    """Call ``generate_fn``, run :func:`detect`, retry up to ``max_retries`` if flagged.

    Use when you want best-effort LLM regeneration with hallucination-informed retry.

    The initial call is ``generate_fn(prompt)`` with no feedback kwarg. On
    flagged retries, the next call is ``generate_fn(prompt, feedback=<str>)``
    where ``<str>`` is built from ``result.top_regime`` per ``feedback_detail``.

    Parameters
    ----------
    generate_fn : Callable[..., str]
        Your LLM-call function. Signature: ``(prompt, feedback=None) -> str``.
        On the initial call, ``feedback`` is not passed; on retries (when
        ``feedback_detail != 'none'``), it is.
    prompt : str
        The prompt to pass to ``generate_fn`` (and to :func:`detect`).
    max_retries : int
        Maximum retries after the initial call. ``max_retries=0`` means a
        single attempt with no retry.
    acceptance_threshold : float
        User-side acceptance threshold. ``result.p_hallucination <= threshold``
        is accepted. Defaults to ``0.5``.
    on_exhausted : str
        Behavior when retries are exhausted without acceptance.

        - ``"raise"`` (default): raise :class:`HaluRegenerationExhausted`.
        - ``"return_best"``: return the attempt with the lowest
          ``p_hallucination``.
        - ``"return_last"``: return the last attempt.
    feedback_detail : str
        Feedback strategy on retries.

        - ``"regime"`` (default): regime-specific hint from
          :data:`_REGIME_GUIDANCE` (falls back to ``"binary"`` if the
          server returns an unknown regime).
        - ``"binary"``: a generic "previous output was flagged" hint.
        - ``"none"``: blind retry — no feedback kwarg passed.
    **detect_kwargs
        Forwarded to :func:`detect` on every iteration: ``task``, ``api_key``,
        ``base_url``, ``timeout``.

    Returns
    -------
    tuple[str, DetectResult, list[tuple[str, DetectResult]]]
        ``(accepted_response, accepted_result, history)`` where ``history``
        is every ``(response, result)`` pair in order, including the
        accepted one.

    Raises
    ------
    HaluRegenerationExhausted
        Retries exhausted with ``on_exhausted="raise"``.
    HaluError
        Any error from :func:`detect` propagates unchanged (the regeneration
        loop does not swallow these).
    ValueError
        ``on_exhausted`` or ``feedback_detail`` is not a recognized value.
    """
    if on_exhausted not in ("raise", "return_best", "return_last"):
        raise ValueError(
            f"on_exhausted must be 'raise', 'return_best', or 'return_last' "
            f"(got {on_exhausted!r})."
        )
    if feedback_detail not in ("regime", "binary", "none"):
        raise ValueError(
            f"feedback_detail must be 'regime', 'binary', or 'none' "
            f"(got {feedback_detail!r})."
        )
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0 (got {max_retries}).")

    history: List[Tuple[str, DetectResult]] = []
    response = generate_fn(prompt)

    total_attempts = max_retries + 1
    for attempt_idx in range(total_attempts):
        result = detect(response, prompt=prompt, **detect_kwargs)
        history.append((response, result))

        if result.p_hallucination <= acceptance_threshold:
            return response, result, history

        is_last = attempt_idx == total_attempts - 1
        if is_last:
            break

        feedback = _build_feedback(result, feedback_detail)
        if feedback is None:
            response = generate_fn(prompt)
        else:
            response = generate_fn(prompt, feedback=feedback)

    if on_exhausted == "raise":
        raise HaluRegenerationExhausted(
            message=(
                f"Regeneration exhausted after {total_attempts} attempt(s) "
                f"without reaching acceptance_threshold={acceptance_threshold} "
                f"(best p_hallucination={min(r.p_hallucination for _, r in history):.3f})."
            ),
            history=history,
        )
    if on_exhausted == "return_best":
        best_response, best_result = min(history, key=lambda pair: pair[1].p_hallucination)
        return best_response, best_result, history
    last_response, last_result = history[-1]
    return last_response, last_result, history
