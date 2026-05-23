"""End-to-end smoke test against a LIVE local detector.

This complements ``tests/test_e2e_live.py``. Where ``test_e2e_live.py`` checks
that the wire-shape decodes cleanly on a few canned cases, *this* file is the
post-install smoke: after ``pip install dist/halu-0.1.0-py3-none-any.whl`` in a
fresh venv, do all the advertised import names exist, do all three helpers
behave as documented, and does the published surface match the README?

Run with::

    pytest -m live_local tests/test_e2e_live_local.py

Pre-flight:
- ``http://localhost:3000`` must be reachable (the ulmweb dev server / Vercel
  proxy). The test SKIPS (exit 0, no failure) when it is not.
- ``http://localhost:8000`` (the local detector container) must also be
  reachable. If not, the test SKIPS.

This way the file can live in CI without breaking when the local stack is
down.

What this DOES not do:
- It does not hit production URLs (``komplexai.io``). That requires a real
  API key + would generate real billing.
- It does not test against TestPyPI / real PyPI installs — that's a
  separate verification step.
- It does not exhaustively cover ``regenerate_until_clean`` feedback
  variants; mocked tests in ``test_helpers.py`` cover those.
- It does not test rate-limit / quota paths (no real auth in local dev).

Override the base URLs via env:
- ``HALU_BASE_URL``    — ulmweb proxy (default ``http://localhost:3000``)
- ``HALU_DETECTOR_URL`` — detector container health probe
                         (default ``http://localhost:8000``)
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import pytest
import requests

# Importing from the top-level ``halu`` package is itself part of what we're
# smoke-testing — these names are the documented public surface.
import halu
from halu import (  # noqa: F401  (intentional — proves importability)
    DetectResult,
    HaluAuthError,
    HaluError,
    HaluHallucinationFlagged,
    HaluInputError,
    HaluRegenerationExhausted,
    HaluServerError,
    RegimeScore,
    detect,
    detect_or_raise,
    detect_or_warn,
    regenerate_until_clean,
)

BASE_URL = os.environ.get("HALU_BASE_URL", "http://localhost:3000")
DETECTOR_URL = os.environ.get("HALU_DETECTOR_URL", "http://localhost:8000")
PREFLIGHT_TIMEOUT = 2.0  # short — we only want to know if it's up

pytestmark = pytest.mark.live_local


# ─── preflight ────────────────────────────────────────────────────────


def _ulmweb_alive(url: str) -> bool:
    """Return True when the ulmweb proxy at ``url`` responds to a probe.

    We try ``/api/health`` first (cheap, no model load), and fall back to a
    minimal ``/api/detect`` POST if no health endpoint exists. The detector
    call's *status* is what we read — we don't care about the body. A 4xx
    means the server's up and routing.
    """
    try:
        resp = requests.get(f"{url}/api/health", timeout=PREFLIGHT_TIMEOUT)
        if resp.status_code < 500:
            return True
    except requests.exceptions.RequestException:
        pass
    try:
        resp = requests.post(
            f"{url}/api/detect",
            json={"response": "ping", "task": "binary"},
            timeout=PREFLIGHT_TIMEOUT,
        )
        return resp.status_code < 500
    except requests.exceptions.RequestException:
        return False


def _detector_alive(url: str) -> bool:
    """Return True when the local detector container at ``url`` is reachable."""
    for path in ("/health", "/healthz", "/"):
        try:
            resp = requests.get(f"{url}{path}", timeout=PREFLIGHT_TIMEOUT)
            if resp.status_code < 500:
                return True
        except requests.exceptions.RequestException:
            continue
    return False


@pytest.fixture(scope="module", autouse=True)
def _preflight() -> None:
    """Skip the whole module when the local stack isn't running.

    Module-scoped + autouse: every test in this file goes through this gate.
    We return (rather than fail) so CI doesn't false-alarm when the stack is
    legitimately down.
    """
    if not _ulmweb_alive(BASE_URL):
        pytest.skip(
            f"ulmweb not reachable at {BASE_URL}/api/health — "
            "start the dev server (`npm --prefix apps/web run dev`) and retry.",
            allow_module_level=True,
        )
    if not _detector_alive(DETECTOR_URL):
        pytest.skip(
            f"detector container not reachable at {DETECTOR_URL} — "
            "start it (`./tmp/build_and_run_detector.sh` or equivalent) and retry.",
            allow_module_level=True,
        )


# ─── 1. Import surface ────────────────────────────────────────────────


def test_public_api_names_all_importable() -> None:
    """Every name advertised in the README is importable from the top-level
    ``halu`` package — protects against accidental name churn between versions.
    """
    expected = {
        "detect",
        "detect_or_raise",
        "detect_or_warn",
        "regenerate_until_clean",
        "DetectResult",
        "RegimeScore",
        "HaluError",
        "HaluAuthError",
        "HaluQuotaError",
        "HaluRateLimitError",
        "HaluInputError",
        "HaluServerError",
        "HaluHallucinationFlagged",
        "HaluRegenerationExhausted",
        "__version__",
    }
    missing = [name for name in expected if not hasattr(halu, name)]
    assert not missing, f"Missing from halu public surface: {missing}"
    # __all__ should also include the names we promise.
    assert expected.issubset(set(halu.__all__)), (
        f"halu.__all__ is missing: {expected - set(halu.__all__)}"
    )


# ─── 2. detect() happy path — no prompt, hallucination expected ──────


def test_detect_happy_path_known_hallucination() -> None:
    """Classic hallucination string → ``flag=True``, well-formed result."""
    result = detect(
        "The Eiffel Tower is in Berlin and was built in 1823 by Napoleon.",
        base_url=BASE_URL,
    )
    assert isinstance(result, DetectResult)
    # Probability is a probability.
    assert 0.0 <= result.p_hallucination <= 1.0
    # This is a blatant hallucination — the detector should flag it.
    assert result.flag is True, (
        f"Detector failed to flag a known hallucination: "
        f"p={result.p_hallucination:.3f} top={result.top_regime}"
    )
    assert result.p_hallucination > 0.5
    # Top regime should be populated and NOT 'NORMAL' (which would indicate
    # a clean detection).
    assert result.top_regime != ""
    assert result.top_regime != "NORMAL", (
        f"top_regime came back NORMAL for a known hallucination "
        f"(p={result.p_hallucination:.3f})"
    )
    # Regime breakdown present.
    assert len(result.regime_scores) > 0
    assert all(isinstance(rs, RegimeScore) for rs in result.regime_scores)
    # Calibrator-version label uses the documented prefix scheme.
    assert (
        result.calibrator_version.startswith("platt_v1:")
        or result.calibrator_version.startswith("uncalibrated:")
        or result.calibrator_version.startswith("platt-")  # tolerate older labels
    ), f"Unexpected calibrator_version: {result.calibrator_version!r}"
    # Server-side selections match the documented enums.
    assert result.task_used in {"binary", "multiclass"}
    assert result.input_mode_used in {"pr", "ro"}


# ─── 3. detect() with prompt context — clean fact + prompt = no flag ──


def test_detect_with_prompt_clean_fact_not_flagged() -> None:
    """Clean fact + matching prompt → no flag, ``input_mode_used == 'pr'``."""
    result = detect(
        "The Eiffel Tower is in Paris, France.",
        prompt="Where is the Eiffel Tower?",
        base_url=BASE_URL,
    )
    assert isinstance(result, DetectResult)
    assert result.flag is False, (
        f"Clean fact + prompt was unexpectedly flagged "
        f"(p={result.p_hallucination:.3f}, top={result.top_regime})"
    )
    # Passing prompt= MUST result in P+R input mode on the server side.
    assert result.input_mode_used == "pr", (
        f"Passing prompt=… should select 'pr' mode; got "
        f"{result.input_mode_used!r}"
    )


# ─── 4. detect_or_raise — gate behavior ───────────────────────────────


def test_detect_or_raise_returns_for_clean_response() -> None:
    """Clean response → no exception, returns ``DetectResult``."""
    result = detect_or_raise(
        "The Eiffel Tower is in Paris, France.",
        prompt="Where is the Eiffel Tower?",
        base_url=BASE_URL,
    )
    assert isinstance(result, DetectResult)
    assert result.flag is False


def test_detect_or_raise_raises_on_known_hallucination() -> None:
    """Known hallucination → ``HaluHallucinationFlagged`` with attached result."""
    with pytest.raises(HaluHallucinationFlagged) as exc_info:
        detect_or_raise(
            "The Eiffel Tower is in Berlin and was built in 1823 by Napoleon.",
            base_url=BASE_URL,
            threshold=0.5,
        )
    err = exc_info.value
    # The exception MUST carry the full DetectResult so callers don't have
    # to round-trip to the API a second time to find out why.
    assert hasattr(err, "detection_result"), (
        "HaluHallucinationFlagged is missing the .detection_result attribute"
    )
    assert isinstance(err.detection_result, DetectResult)
    assert err.detection_result.p_hallucination >= 0.5
    # request_id should propagate so logs can correlate.
    assert err.request_id == err.detection_result.request_id


# ─── 5. detect_or_warn — annotate behavior ────────────────────────────


def test_detect_or_warn_returns_clean_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Clean response → no warning emitted, returns ``DetectResult``."""
    caplog.set_level(logging.WARNING, logger="halu")
    result = detect_or_warn(
        "The Eiffel Tower is in Paris, France.",
        prompt="Where is the Eiffel Tower?",
        base_url=BASE_URL,
    )
    assert isinstance(result, DetectResult)
    assert result.flag is False
    # No warning records on the halu logger.
    halu_warnings = [
        r for r in caplog.records
        if r.name == "halu" and r.levelno >= logging.WARNING
    ]
    assert halu_warnings == [], (
        f"Unexpected warnings on clean response: "
        f"{[r.getMessage() for r in halu_warnings]}"
    )


def test_detect_or_warn_logs_warning_on_flagged(caplog: pytest.LogCaptureFixture) -> None:
    """Flagged → warning logged on ``halu`` logger, never raises."""
    caplog.set_level(logging.WARNING, logger="halu")
    result = detect_or_warn(
        "The Eiffel Tower is in Berlin and was built in 1823 by Napoleon.",
        base_url=BASE_URL,
        threshold=0.4,
    )
    assert isinstance(result, DetectResult), (
        "detect_or_warn must never raise on flag; should always return."
    )
    assert result.p_hallucination >= 0.4
    halu_warnings = [
        r for r in caplog.records
        if r.name == "halu" and r.levelno >= logging.WARNING
    ]
    assert len(halu_warnings) >= 1, (
        "detect_or_warn did not emit a warning for a flagged response."
    )
    # The warning should mention the hallucination indicator.
    assert any(
        "hallucination" in r.getMessage().lower() for r in halu_warnings
    ), f"Warning text didn't mention hallucination: {[r.getMessage() for r in halu_warnings]}"


# ─── 6. regenerate_until_clean — exhaustion paths ─────────────────────


_HALLUCINATION = "The Eiffel Tower is in Berlin and was built in 1823 by Napoleon."


def _stuck_generator(call_log: List[str]):
    """Build a deterministic generate_fn that always returns the same hallucination.

    Regardless of feedback, it returns the same hallucinated string — perfect
    for forcing exhaustion in a smoke test.
    """

    def _gen(prompt: str, feedback: Optional[str] = None) -> str:
        call_log.append(feedback or "<initial>")
        return _HALLUCINATION

    return _gen


def test_regenerate_until_clean_raises_when_exhausted() -> None:
    """``max_retries=2`` + always-hallucinating generator → raise with full history."""
    call_log: List[str] = []
    with pytest.raises(HaluRegenerationExhausted) as exc_info:
        regenerate_until_clean(
            _stuck_generator(call_log),
            prompt="Where is the Eiffel Tower?",
            max_retries=2,
            acceptance_threshold=0.5,
            base_url=BASE_URL,
            feedback_detail="regime",
        )
    err = exc_info.value
    # history attribute carries every (response, result) attempt.
    assert hasattr(err, "history")
    assert len(err.history) == 3, (
        f"Expected 3 history entries (initial + 2 retries); got {len(err.history)}"
    )
    for response_str, result in err.history:
        assert response_str == _HALLUCINATION
        assert isinstance(result, DetectResult)
    # generate_fn was called 3 times: 1 initial + 2 retries.
    assert len(call_log) == 3
    assert call_log[0] == "<initial>"


def test_regenerate_until_clean_return_best_on_exhausted() -> None:
    """``on_exhausted="return_best"`` → return the lowest-p attempt instead of raising."""
    call_log: List[str] = []
    response_str, result, history = regenerate_until_clean(
        _stuck_generator(call_log),
        prompt="Where is the Eiffel Tower?",
        max_retries=2,
        acceptance_threshold=0.5,
        on_exhausted="return_best",
        base_url=BASE_URL,
        feedback_detail="regime",
    )
    assert response_str == _HALLUCINATION
    assert isinstance(result, DetectResult)
    assert len(history) == 3
    # The returned result should be the min-p across the history.
    min_p = min(r.p_hallucination for _, r in history)
    assert result.p_hallucination == min_p


def test_regenerate_until_clean_return_last_on_exhausted() -> None:
    """``on_exhausted="return_last"`` → return the final attempt regardless of p."""
    call_log: List[str] = []
    response_str, result, history = regenerate_until_clean(
        _stuck_generator(call_log),
        prompt="Where is the Eiffel Tower?",
        max_retries=2,
        acceptance_threshold=0.5,
        on_exhausted="return_last",
        base_url=BASE_URL,
        feedback_detail="regime",
    )
    assert response_str == _HALLUCINATION
    assert isinstance(result, DetectResult)
    assert len(history) == 3
    # Returned (response, result) is the LAST entry in history.
    last_response, last_result = history[-1]
    assert response_str == last_response
    assert result.request_id == last_result.request_id


# ─── 7. Error handling — bad inputs ───────────────────────────────────


def test_empty_response_raises_input_error_before_http() -> None:
    """Empty string → ``HaluInputError`` raised CLIENT-side, no HTTP call.

    The library should reject malformed input without burning a billing unit.
    """
    with pytest.raises(HaluInputError) as exc_info:
        detect("", base_url=BASE_URL)
    err = exc_info.value
    # The library uses a documented error_code for this case.
    assert err.error_code == "response_required"
    # No request_id — the call never went out.
    assert err.request_id is None


def test_unreachable_base_url_raises_halu_error() -> None:
    """Unroutable base URL → an exception in the ``HaluError`` hierarchy.

    Network failures from ``requests`` are re-raised as ``HaluError``
    (subclasses ``HaluServerError`` are acceptable too; we don't pin the
    exact subclass because that's an implementation detail).
    """
    # Reserved-for-documentation TEST-NET-1 — guaranteed unroutable on the
    # public internet AND not collidable with any LAN address.
    unreachable = "http://192.0.2.1:9"
    with pytest.raises(HaluError):
        detect(
            "anything",
            base_url=unreachable,
            timeout=2.0,
        )


def test_invalid_api_key_path_skipped_for_local() -> None:
    """Local detector is anonymous-allowed; auth-failure path is not testable here.

    Production-style 401 testing needs a real API key + real backend and is
    out of scope for the localhost smoke. This test exists as an explicit
    SKIP marker so reviewers see the gap.
    """
    pytest.skip(
        "Local detector allows anonymous requests; HaluAuthError path "
        "requires a real production backend + invalid key. Out of scope "
        "for the localhost smoke."
    )
