"""Tests for halu.detect() and the HTTP client."""
from __future__ import annotations

from typing import Any, Dict

import pytest
import responses

import halu
from halu import (
    DetectResult,
    HaluAuthError,
    HaluError,
    HaluInputError,
    HaluQuotaError,
    HaluRateLimitError,
    HaluServerError,
    RegimeScore,
    detect,
)

LOCAL_URL = "http://localhost:3000"
PROD_URL = "https://komplexai.io"
DETECT_LOCAL = f"{LOCAL_URL}/api/detect"
DETECT_PROD = f"{PROD_URL}/api/detect"


@responses.activate
def test_detect_happy_path_response_only(detect_response_ok: Dict[str, Any]) -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        json=detect_response_ok,
        status=200,
    )
    result = detect("Plain factual sentence.", base_url=LOCAL_URL)

    assert isinstance(result, DetectResult)
    assert result.p_hallucination == pytest.approx(0.12)
    assert result.flag is False
    assert result.top_regime == "NORMAL"
    assert result.task_used == "multiclass"
    assert result.input_mode_used == "ro"
    assert isinstance(result.regime_scores, list)
    assert isinstance(result.regime_scores[0], RegimeScore)

    sent = responses.calls[0].request
    body = sent.body
    assert b"\"response\"" in body
    assert b"\"prompt\"" not in body
    assert b"\"task\"" in body


@responses.activate
def test_detect_with_prompt_includes_prompt_in_body(detect_response_flagged: Dict[str, Any]) -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        json=detect_response_flagged,
        status=200,
    )
    result = detect(
        response="The Eiffel Tower was built in Berlin.",
        prompt="Where is the Eiffel Tower?",
        base_url=LOCAL_URL,
    )

    assert result.flag is True
    assert result.top_regime == "FABRICATED"
    body = responses.calls[0].request.body
    assert b"Where is the Eiffel Tower" in body


@responses.activate
def test_detect_task_binary(detect_response_ok: Dict[str, Any]) -> None:
    body = dict(detect_response_ok)
    body["task_used"] = "binary"
    body["regime_scores"] = [{"regime": "binary_hallucination", "p": 0.12}]
    responses.add(responses.POST, DETECT_LOCAL, json=body, status=200)

    result = detect("text", task="binary", base_url=LOCAL_URL)
    assert result.task_used == "binary"
    assert result.regime_scores[0].regime == "binary_hallucination"

    sent_body = responses.calls[0].request.body
    assert b'"task": "binary"' in sent_body or b'"task":"binary"' in sent_body


@responses.activate
def test_detect_to_dict_round_trip(detect_response_ok: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)
    result = detect("text", base_url=LOCAL_URL)
    d = result.to_dict()
    assert d["p_hallucination"] == pytest.approx(0.12)
    assert d["regime_scores"][0] == {"regime": "NORMAL", "p": 0.72}
    assert "raw" not in d


def test_detect_empty_response_raises_input_error() -> None:
    with pytest.raises(HaluInputError) as exc_info:
        detect("", base_url=LOCAL_URL)
    assert exc_info.value.error_code == "response_required"


def test_detect_invalid_task_raises_input_error() -> None:
    with pytest.raises(HaluInputError):
        detect("text", task="not-a-task", base_url=LOCAL_URL)  # type: ignore[arg-type]


def test_detect_missing_key_for_remote_raises_auth_error() -> None:
    with pytest.raises(HaluAuthError) as exc_info:
        detect("text", base_url="https://example.com")
    assert "HALU_API_KEY" in exc_info.value.message


@responses.activate
def test_detect_localhost_skips_auth(detect_response_ok: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)
    result = detect("text", base_url=LOCAL_URL)
    assert isinstance(result, DetectResult)
    sent = responses.calls[0].request
    assert "Authorization" not in sent.headers


@responses.activate
def test_detect_sends_bearer_header(detect_response_ok: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_PROD, json=detect_response_ok, status=200)
    detect("text", api_key="sk_test_abc", base_url=PROD_URL)
    sent = responses.calls[0].request
    assert sent.headers["Authorization"] == "Bearer sk_test_abc"


@responses.activate
def test_detect_api_key_from_env(
    detect_response_ok: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HALU_API_KEY", "sk_env_key_xyz")
    responses.add(responses.POST, DETECT_PROD, json=detect_response_ok, status=200)
    detect("text", base_url=PROD_URL)
    sent = responses.calls[0].request
    assert sent.headers["Authorization"] == "Bearer sk_env_key_xyz"


@responses.activate
def test_detect_base_url_from_env(
    detect_response_ok: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HALU_BASE_URL", LOCAL_URL)
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)
    result = detect("text")
    assert isinstance(result, DetectResult)


@responses.activate
def test_detect_timeout_from_env(
    detect_response_ok: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HALU_TIMEOUT", "7")
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)

    from halu import _client as client_module
    real_post = client_module.requests.post
    captured: Dict[str, Any] = {}

    def spy_post(*args: Any, **kwargs: Any) -> Any:
        captured["timeout"] = kwargs.get("timeout")
        return real_post(*args, **kwargs)

    monkeypatch.setattr(client_module.requests, "post", spy_post)
    detect("text", base_url=LOCAL_URL)
    assert captured["timeout"] == pytest.approx(7.0)


@responses.activate
def test_detect_strips_trailing_slash_from_base_url(detect_response_ok: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)
    result = detect("text", base_url=LOCAL_URL + "/")
    assert isinstance(result, DetectResult)


@responses.activate
def test_detect_401_raises_auth_error() -> None:
    responses.add(
        responses.POST,
        DETECT_PROD,
        json={"error": "unauthorized", "message": "bad key", "request_id": "req_xyz"},
        status=401,
    )
    with pytest.raises(HaluAuthError) as exc_info:
        detect("text", api_key="sk_bad", base_url=PROD_URL)
    assert exc_info.value.request_id == "req_xyz"
    assert "bad key" in exc_info.value.message


@responses.activate
def test_detect_402_raises_quota_error() -> None:
    responses.add(
        responses.POST,
        DETECT_PROD,
        json={
            "error": "quota_exceeded",
            "message": "Free quota exhausted",
            "upgrade_url": "https://komplexai.io/account/billing",
            "request_id": "req_quota",
        },
        status=402,
        headers={"X-Quota-Period": "2026-05"},
    )
    with pytest.raises(HaluQuotaError) as exc_info:
        detect("text", api_key="sk_x", base_url=PROD_URL)
    err = exc_info.value
    assert err.quota_period == "2026-05"
    assert err.upgrade_url == "https://komplexai.io/account/billing"
    assert err.request_id == "req_quota"


@responses.activate
def test_detect_402_accepts_detail_shape() -> None:
    responses.add(
        responses.POST,
        DETECT_PROD,
        json={"detail": "quota_exceeded"},
        status=402,
    )
    with pytest.raises(HaluQuotaError):
        detect("text", api_key="sk_x", base_url=PROD_URL)


@responses.activate
def test_detect_429_raises_rate_limit_error() -> None:
    responses.add(
        responses.POST,
        DETECT_PROD,
        json={"error": "rate_limited", "message": "slow down"},
        status=429,
        headers={"Retry-After": "5"},
    )
    with pytest.raises(HaluRateLimitError) as exc_info:
        detect("text", api_key="sk_x", base_url=PROD_URL)
    assert exc_info.value.retry_after == 5


@responses.activate
def test_detect_429_missing_retry_after_returns_none() -> None:
    responses.add(
        responses.POST,
        DETECT_PROD,
        json={"error": "rate_limited"},
        status=429,
    )
    with pytest.raises(HaluRateLimitError) as exc_info:
        detect("text", api_key="sk_x", base_url=PROD_URL)
    assert exc_info.value.retry_after is None


@responses.activate
def test_detect_400_raises_input_error() -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        json={"error": "bad_json", "message": "could not parse"},
        status=400,
    )
    with pytest.raises(HaluInputError) as exc_info:
        detect("text", base_url=LOCAL_URL)
    assert exc_info.value.error_code == "bad_json"


@responses.activate
def test_detect_500_raises_server_error() -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        json={"error": "handler_failed", "request_id": "req_500"},
        status=500,
    )
    with pytest.raises(HaluServerError) as exc_info:
        detect("text", base_url=LOCAL_URL)
    assert exc_info.value.status_code == 500
    assert exc_info.value.request_id == "req_500"


@responses.activate
def test_detect_502_raises_server_error_with_detail() -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        json={"detail": "upstream timeout"},
        status=502,
    )
    with pytest.raises(HaluServerError) as exc_info:
        detect("text", base_url=LOCAL_URL)
    assert exc_info.value.status_code == 502
    assert exc_info.value.upstream_message == "upstream timeout"


@responses.activate
def test_detect_malformed_200_body_raises_server_error() -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        body="not-json",
        status=200,
        content_type="application/json",
    )
    with pytest.raises(HaluServerError):
        detect("text", base_url=LOCAL_URL)


@responses.activate
def test_detect_200_missing_required_field_raises_server_error() -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        json={"p_hallucination": "not-a-number", "regime_scores": []},
        status=200,
    )
    with pytest.raises(HaluServerError):
        detect("text", base_url=LOCAL_URL)


@responses.activate
def test_detect_request_id_header_populated_on_error() -> None:
    responses.add(
        responses.POST,
        DETECT_LOCAL,
        json={"error": "bad_json"},
        status=400,
        headers={"X-Request-Id": "req_from_header"},
    )
    with pytest.raises(HaluInputError) as exc_info:
        detect("text", base_url=LOCAL_URL)
    assert exc_info.value.request_id == "req_from_header"


def test_detect_timeout_raises_halu_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from halu import _client as client_module

    def fake_post(*args: Any, **kwargs: Any) -> Any:
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    with pytest.raises(HaluError) as exc_info:
        detect("text", base_url=LOCAL_URL, timeout=1)
    assert "timed out" in exc_info.value.message.lower()


def test_detect_network_error_raises_halu_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from halu import _client as client_module

    def fake_post(*args: Any, **kwargs: Any) -> Any:
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    with pytest.raises(HaluError) as exc_info:
        detect("text", base_url=LOCAL_URL)
    assert "no route" in exc_info.value.message


@responses.activate
def test_detect_sends_user_agent_header(detect_response_ok: Dict[str, Any]) -> None:
    responses.add(responses.POST, DETECT_LOCAL, json=detect_response_ok, status=200)
    detect("text", base_url=LOCAL_URL)
    ua = responses.calls[0].request.headers.get("User-Agent", "")
    assert ua.startswith("halu-python/")


def test_version_is_pep440_string() -> None:
    assert halu.__version__ == "0.1.0"
