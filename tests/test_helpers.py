"""Tests for the three helper functions: detect_or_raise / detect_or_warn / regenerate_until_clean."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytest
import responses

from halu import (
    DetectResult,
    HaluHallucinationFlagged,
    HaluRateLimitError,
    HaluRegenerationExhausted,
    detect_or_raise,
    detect_or_warn,
    regenerate_until_clean,
)
from halu._helpers import _BINARY_FEEDBACK, _REGIME_GUIDANCE, _build_feedback

LOCAL_URL = "http://localhost:3000"
DETECT_LOCAL = f"{LOCAL_URL}/api/detect"


def _flagged_body(p: float = 0.87, top_regime: str = "FABRICATED") -> Dict[str, Any]:
    return {
        "p_hallucination": p,
        "flag": p >= 0.5,
        "top_regime": top_regime,
        "regime_scores": [{"regime": top_regime, "p": p}],
        "request_id": f"req_{top_regime.lower()}_{int(p * 100)}",
        "detections_billed": 1,
        "mode": "short",
        "latency_ms": 100,
        "model_version": "nl-v1",
        "calibrator_version": "platt-v3",
        "input_mode_used": "ro",
        "task_used": "multiclass",
    }


# ─── detect_or_raise ─────────────────────────────────────────────────


@responses.activate
def test_detect_or_raise_returns_when_not_flagged(detect_response_ok: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)
    result = detect_or_raise("clean text", base_url=LOCAL_URL)
    assert isinstance(result, DetectResult)
    assert result.p_hallucination < 0.5


@responses.activate
def test_detect_or_raise_raises_when_flagged(detect_response_flagged: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    with pytest.raises(HaluHallucinationFlagged) as exc_info:
        detect_or_raise("bad text", base_url=LOCAL_URL)
    err = exc_info.value
    assert err.detection_result.top_regime == "FABRICATED"
    assert err.detection_result.p_hallucination == pytest.approx(0.87)
    assert err.request_id == "req_test_flagged_0002"


@responses.activate
def test_detect_or_raise_threshold_respected_strict(detect_response_ok: Dict[str, Any]) -> None:
    body = dict(detect_response_ok)
    body["p_hallucination"] = 0.30
    responses.add(responses.POST, DETECT_LOCAL, json=body, status=200)
    with pytest.raises(HaluHallucinationFlagged):
        detect_or_raise("text", threshold=0.20, base_url=LOCAL_URL)


@responses.activate
def test_detect_or_raise_threshold_respected_loose(detect_response_flagged: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    result = detect_or_raise("text", threshold=0.99, base_url=LOCAL_URL)
    assert result.p_hallucination == pytest.approx(0.87)


@responses.activate
def test_detect_or_raise_raise_on_flag_false_returns(
    detect_response_flagged: Dict[str, Any],
) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    result = detect_or_raise(
        "text",
        threshold=0.5,
        raise_on_flag=False,
        base_url=LOCAL_URL,
    )
    assert result.flag is True


@responses.activate
def test_detect_or_raise_forwards_prompt_and_task(detect_response_flagged: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    with pytest.raises(HaluHallucinationFlagged):
        detect_or_raise(
            "bad text",
            prompt="why?",
            task="binary",
            base_url=LOCAL_URL,
        )
    body = responses.calls[0].request.body
    assert b"why?" in body
    assert b"binary" in body


# ─── detect_or_warn ──────────────────────────────────────────────────


@responses.activate
def test_detect_or_warn_no_warning_when_not_flagged(
    detect_response_ok: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)
    with caplog.at_level(logging.WARNING, logger="halu"):
        result = detect_or_warn("clean", base_url=LOCAL_URL)
    assert isinstance(result, DetectResult)
    assert len(caplog.records) == 0


@responses.activate
def test_detect_or_warn_warns_when_flagged(
    detect_response_flagged: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    with caplog.at_level(logging.WARNING, logger="halu"):
        result = detect_or_warn("bad", base_url=LOCAL_URL)
    assert result.flag is True
    assert any("hallucination" in rec.getMessage().lower() for rec in caplog.records)


@responses.activate
def test_detect_or_warn_threshold_lower_default(detect_response_ok: Dict[str, Any]) -> None:
    body = dict(detect_response_ok)
    body["p_hallucination"] = 0.45
    responses.add(responses.POST, DETECT_LOCAL, json=body, status=200)
    logger = logging.getLogger("test_detect_or_warn_lower")
    records: List[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

    try:
        detect_or_warn("text", logger=logger, base_url=LOCAL_URL)
    finally:
        logger.removeHandler(handler)
    assert len(records) == 1


@responses.activate
def test_detect_or_warn_custom_logger(detect_response_flagged: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    custom = logging.getLogger("custom_halu_test")
    records: List[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    custom.addHandler(handler)
    custom.setLevel(logging.WARNING)
    custom.propagate = False

    try:
        detect_or_warn("text", logger=custom, base_url=LOCAL_URL)
    finally:
        custom.removeHandler(handler)
    assert len(records) == 1
    assert "hallucination" in records[0].getMessage().lower()


@responses.activate
def test_detect_or_warn_default_logger_used(
    detect_response_flagged: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    with caplog.at_level(logging.WARNING, logger="halu"):
        detect_or_warn("text", base_url=LOCAL_URL)
    halu_records = [r for r in caplog.records if r.name == "halu"]
    assert len(halu_records) == 1


@responses.activate
def test_detect_or_warn_returns_even_when_flagged(
    detect_response_flagged: Dict[str, Any],
) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    result = detect_or_warn("text", base_url=LOCAL_URL)
    assert isinstance(result, DetectResult)


# ─── _build_feedback unit tests ──────────────────────────────────────


def _bare_result(top_regime: str, p: float = 0.9) -> DetectResult:
    return DetectResult(
        p_hallucination=p,
        flag=True,
        top_regime=top_regime,
        regime_scores=[],
        request_id="req_t",
        detections_billed=1,
        mode="short",
        latency_ms=10,
        model_version="nl-v1",
        calibrator_version="platt-v3",
        input_mode_used="ro",
        task_used="multiclass",
    )


def test_build_feedback_none_returns_none() -> None:
    assert _build_feedback(_bare_result("FABRICATED"), "none") is None


def test_build_feedback_binary_returns_static_string() -> None:
    assert _build_feedback(_bare_result("FABRICATED"), "binary") == _BINARY_FEEDBACK


def test_build_feedback_regime_known() -> None:
    msg = _build_feedback(_bare_result("FABRICATED"), "regime")
    assert msg is not None
    assert "FABRICATED" in msg
    assert _REGIME_GUIDANCE["FABRICATED"] in msg


def test_build_feedback_regime_all_known_codes() -> None:
    for code in _REGIME_GUIDANCE.keys():
        msg = _build_feedback(_bare_result(code), "regime")
        assert msg is not None
        assert code in msg
        assert _REGIME_GUIDANCE[code] in msg


def test_build_feedback_regime_unknown_falls_back_to_binary() -> None:
    msg = _build_feedback(_bare_result("UNKNOWN_REGIME"), "regime")
    assert msg == _BINARY_FEEDBACK


def test_build_feedback_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        _build_feedback(_bare_result("FABRICATED"), "bogus")


# ─── regenerate_until_clean ──────────────────────────────────────────


@responses.activate
def test_regenerate_accepts_on_first_call(detect_response_ok: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)

    call_count = {"n": 0}

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        call_count["n"] += 1
        return f"answer for {prompt}"

    response, result, history = regenerate_until_clean(
        gen, "the question", base_url=LOCAL_URL,
    )
    assert response == "answer for the question"
    assert isinstance(result, DetectResult)
    assert len(history) == 1
    assert call_count["n"] == 1


@responses.activate
def test_regenerate_accepts_on_retry(
    detect_response_flagged: Dict[str, Any],
    detect_response_ok: Dict[str, Any],
) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)

    feedbacks: List[Optional[str]] = []

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        feedbacks.append(feedback)
        return f"v{len(feedbacks)}"

    response, result, history = regenerate_until_clean(
        gen, "q", max_retries=2, base_url=LOCAL_URL,
    )
    assert response == "v2"
    assert result.flag is False
    assert len(history) == 2
    assert feedbacks[0] is None
    assert feedbacks[1] is not None
    assert "FABRICATED" in feedbacks[1]


@responses.activate
def test_regenerate_exhausts_and_raises(detect_response_flagged: Dict[str, Any]) -> None:
    for _ in range(3):
        responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        return "still bad"

    with pytest.raises(HaluRegenerationExhausted) as exc_info:
        regenerate_until_clean(gen, "q", max_retries=2, base_url=LOCAL_URL)
    err = exc_info.value
    assert len(err.history) == 3
    assert err.history[0][0] == "still bad"


@responses.activate
def test_regenerate_exhausts_return_best(detect_response_flagged: Dict[str, Any]) -> None:
    bodies = [
        _flagged_body(p=0.9),
        _flagged_body(p=0.55),
        _flagged_body(p=0.75),
    ]
    for b in bodies:
        responses.add(responses.POST, DETECT_LOCAL, json=b, status=200)

    versions: List[str] = []

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        versions.append(f"v{len(versions) + 1}")
        return versions[-1]

    response, result, history = regenerate_until_clean(
        gen,
        "q",
        max_retries=2,
        on_exhausted="return_best",
        base_url=LOCAL_URL,
    )
    assert result.p_hallucination == pytest.approx(0.55)
    assert response == "v2"
    assert len(history) == 3


@responses.activate
def test_regenerate_exhausts_return_last(detect_response_flagged: Dict[str, Any]) -> None:
    bodies = [
        _flagged_body(p=0.9),
        _flagged_body(p=0.55),
        _flagged_body(p=0.75),
    ]
    for b in bodies:
        responses.add(responses.POST, DETECT_LOCAL, json=b, status=200)

    versions: List[str] = []

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        versions.append(f"v{len(versions) + 1}")
        return versions[-1]

    response, result, history = regenerate_until_clean(
        gen,
        "q",
        max_retries=2,
        on_exhausted="return_last",
        base_url=LOCAL_URL,
    )
    assert response == "v3"
    assert result.p_hallucination == pytest.approx(0.75)


@responses.activate
def test_regenerate_feedback_detail_none_no_feedback_kwarg(
    detect_response_flagged: Dict[str, Any],
) -> None:
    bodies = [_flagged_body(p=0.9), _flagged_body(p=0.8)]
    for b in bodies:
        responses.add(responses.POST, DETECT_LOCAL, json=b, status=200)

    call_args: List[Dict[str, Any]] = []

    def gen(prompt: str, **kwargs: Any) -> str:
        call_args.append(kwargs)
        return "still bad"

    with pytest.raises(HaluRegenerationExhausted):
        regenerate_until_clean(
            gen,
            "q",
            max_retries=1,
            feedback_detail="none",
            base_url=LOCAL_URL,
        )
    assert call_args == [{}, {}]


@responses.activate
def test_regenerate_feedback_detail_binary(detect_response_flagged: Dict[str, Any]) -> None:
    bodies = [_flagged_body(p=0.9), _flagged_body(p=0.8)]
    for b in bodies:
        responses.add(responses.POST, DETECT_LOCAL, json=b, status=200)

    feedbacks: List[Optional[str]] = []

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        feedbacks.append(feedback)
        return "still bad"

    with pytest.raises(HaluRegenerationExhausted):
        regenerate_until_clean(
            gen,
            "q",
            max_retries=1,
            feedback_detail="binary",
            base_url=LOCAL_URL,
        )
    assert feedbacks[0] is None
    assert feedbacks[1] == _BINARY_FEEDBACK


@responses.activate
def test_regenerate_feedback_detail_regime(detect_response_flagged: Dict[str, Any]) -> None:
    bodies = [_flagged_body(p=0.9, top_regime="CF_AUTH"), _flagged_body(p=0.8)]
    for b in bodies:
        responses.add(responses.POST, DETECT_LOCAL, json=b, status=200)

    feedbacks: List[Optional[str]] = []

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        feedbacks.append(feedback)
        return "still bad"

    with pytest.raises(HaluRegenerationExhausted):
        regenerate_until_clean(
            gen,
            "q",
            max_retries=1,
            feedback_detail="regime",
            base_url=LOCAL_URL,
        )
    assert feedbacks[0] is None
    assert feedbacks[1] is not None
    assert "CF_AUTH" in feedbacks[1]
    assert _REGIME_GUIDANCE["CF_AUTH"] in feedbacks[1]


@responses.activate
def test_regenerate_max_retries_zero_exhausts_immediately(
    detect_response_flagged: Dict[str, Any],
) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        return "single shot"

    with pytest.raises(HaluRegenerationExhausted) as exc_info:
        regenerate_until_clean(
            gen,
            "q",
            max_retries=0,
            base_url=LOCAL_URL,
        )
    assert len(exc_info.value.history) == 1


@responses.activate
def test_regenerate_propagates_generate_fn_exception(
    detect_response_flagged: Dict[str, Any],
) -> None:
    class BoomError(RuntimeError):
        pass

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        raise BoomError("LLM call failed")

    with pytest.raises(BoomError):
        regenerate_until_clean(gen, "q", base_url=LOCAL_URL)


@responses.activate
def test_regenerate_propagates_detect_exception() -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        json={"error": "rate_limited"},
        status=429,
        headers={"Retry-After": "3"},
    )

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        return "anything"

    with pytest.raises(HaluRateLimitError):
        regenerate_until_clean(gen, "q", base_url=LOCAL_URL)


def test_regenerate_invalid_on_exhausted_raises() -> None:
    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        return "x"

    with pytest.raises(ValueError):
        regenerate_until_clean(gen, "q", on_exhausted="bogus", base_url=LOCAL_URL)


def test_regenerate_invalid_feedback_detail_raises() -> None:
    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        return "x"

    with pytest.raises(ValueError):
        regenerate_until_clean(gen, "q", feedback_detail="bogus", base_url=LOCAL_URL)


def test_regenerate_negative_max_retries_raises() -> None:
    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        return "x"

    with pytest.raises(ValueError):
        regenerate_until_clean(gen, "q", max_retries=-1, base_url=LOCAL_URL)


@responses.activate
def test_regenerate_acceptance_threshold_respected(detect_response_ok: Dict[str, Any]) -> None:
    body = dict(detect_response_ok)
    body["p_hallucination"] = 0.3
    responses.add(responses.POST, DETECT_LOCAL, json=body, status=200)
    responses.add(responses.POST, DETECT_LOCAL, json=body, status=200)

    calls = {"n": 0}

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        calls["n"] += 1
        return "ok"

    with pytest.raises(HaluRegenerationExhausted):
        regenerate_until_clean(
            gen,
            "q",
            max_retries=1,
            acceptance_threshold=0.2,
            base_url=LOCAL_URL,
        )
    assert calls["n"] == 2


@responses.activate
def test_regenerate_history_includes_accepted_attempt(
    detect_response_flagged: Dict[str, Any],
    detect_response_ok: Dict[str, Any],
) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_flagged, status=200)
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)

    def gen(prompt: str, feedback: Optional[str] = None) -> str:
        return "bad" if feedback is None else "good"

    response, result, history = regenerate_until_clean(
        gen, "q", base_url=LOCAL_URL,
    )
    assert response == "good"
    assert len(history) == 2
    assert history[0][1].flag is True
    assert history[1][1].flag is False
