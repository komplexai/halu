"""Tests for the halu exception hierarchy."""
from __future__ import annotations

from halu import (
    DetectResult,
    HaluAuthError,
    HaluError,
    HaluHallucinationFlagged,
    HaluInputError,
    HaluQuotaError,
    HaluRateLimitError,
    HaluRegenerationExhausted,
    HaluServerError,
    RegimeScore,
)


def test_halu_error_is_exception_base() -> None:
    assert issubclass(HaluError, Exception)
    assert issubclass(HaluAuthError, HaluError)
    assert issubclass(HaluQuotaError, HaluError)
    assert issubclass(HaluRateLimitError, HaluError)
    assert issubclass(HaluInputError, HaluError)
    assert issubclass(HaluServerError, HaluError)
    assert issubclass(HaluHallucinationFlagged, HaluError)
    assert issubclass(HaluRegenerationExhausted, HaluError)


def test_halu_error_str_includes_request_id() -> None:
    err = HaluError("boom", request_id="req_x")
    assert "req_x" in str(err)
    assert "boom" in str(err)


def test_halu_error_str_without_request_id() -> None:
    err = HaluError("boom")
    assert str(err) == "boom"


def test_halu_quota_error_attributes() -> None:
    err = HaluQuotaError(
        "quota",
        request_id="req_q",
        quota_period="2026-05",
        upgrade_url="https://example.com",
    )
    assert err.quota_period == "2026-05"
    assert err.upgrade_url == "https://example.com"
    assert err.request_id == "req_q"


def test_halu_rate_limit_error_attributes() -> None:
    err = HaluRateLimitError("slow", retry_after=10)
    assert err.retry_after == 10


def test_halu_input_error_attributes() -> None:
    err = HaluInputError("bad", error_code="bad_json")
    assert err.error_code == "bad_json"


def test_halu_server_error_attributes() -> None:
    err = HaluServerError("oops", status_code=502, upstream_message="bad upstream")
    assert err.status_code == 502
    assert err.upstream_message == "bad upstream"


def _make_result(p: float = 0.9, request_id: str = "req_x") -> DetectResult:
    return DetectResult(
        p_hallucination=p,
        flag=p >= 0.5,
        top_regime="FABRICATED",
        regime_scores=[RegimeScore(regime="FABRICATED", p=0.7)],
        request_id=request_id,
        detections_billed=1,
        mode="short",
        latency_ms=100,
        model_version="nl-v1",
        calibrator_version="platt-v3",
        input_mode_used="pr",
        task_used="multiclass",
    )


def test_halu_hallucination_flagged_carries_detection_result() -> None:
    result = _make_result(p=0.95, request_id="req_flagged")
    err = HaluHallucinationFlagged("flagged", detection_result=result)
    assert err.detection_result.flag is True
    assert err.detection_result.top_regime == "FABRICATED"
    assert err.request_id == "req_flagged"


def test_halu_regeneration_exhausted_carries_history() -> None:
    h = [
        ("first", _make_result(p=0.9, request_id="req_a")),
        ("second", _make_result(p=0.8, request_id="req_b")),
    ]
    err = HaluRegenerationExhausted("exhausted", history=h)
    assert len(err.history) == 2
    assert err.history[0][0] == "first"
    assert err.history[1][1].request_id == "req_b"
    assert err.request_id == "req_b"


def test_halu_regeneration_exhausted_empty_history_ok() -> None:
    err = HaluRegenerationExhausted("exhausted", history=[])
    assert err.history == []
    assert err.request_id is None
