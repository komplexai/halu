"""HTTP client implementation for ``halu.detect``.

Builds the request, calls ``POST {base_url}/api/detect``, and maps the
response (success or error) to a :class:`DetectResult` or the appropriate
:mod:`halu._errors` exception.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

from ._errors import (
    HaluAuthError,
    HaluError,
    HaluInputError,
    HaluQuotaError,
    HaluRateLimitError,
    HaluServerError,
)
from ._types import DetectResult, RegimeScore

DEFAULT_BASE_URL = "https://api.komplexai.io"
DEFAULT_TIMEOUT = 30.0
_DETECT_PATH = "/api/detect"


def _resolve(value: Optional[Any], env_var: str, default: Any) -> Any:
    if value is not None:
        return value
    raw = os.environ.get(env_var)
    if raw is not None and raw != "":
        return raw
    return default


def _is_localhost(base_url: str) -> bool:
    lower = base_url.lower()
    return (
        lower.startswith("http://localhost")
        or lower.startswith("http://127.0.0.1")
        or lower.startswith("http://0.0.0.0")
    )


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _error_payload(body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if body is None:
        return {}
    return body


def _raise_for_status(
    response: "requests.Response",
    request_id: Optional[str],
) -> None:
    status = response.status_code
    if status < 400:
        return

    body = _parse_json(response.text)
    payload = _error_payload(body)
    server_request_id = payload.get("request_id") or request_id
    error_code = payload.get("error")
    detail = payload.get("detail")
    message = payload.get("message") or detail or error_code or response.reason or f"HTTP {status}"

    if status == 401:
        raise HaluAuthError(
            message=message,
            request_id=server_request_id,
        )
    if status == 402:
        raise HaluQuotaError(
            message=message,
            request_id=server_request_id,
            quota_period=response.headers.get("X-Quota-Period"),
            upgrade_url=payload.get("upgrade_url"),
        )
    if status == 429:
        retry_after = _safe_int(response.headers.get("Retry-After"))
        raise HaluRateLimitError(
            message=message,
            request_id=server_request_id,
            retry_after=retry_after,
        )
    if status == 400:
        raise HaluInputError(
            message=message,
            request_id=server_request_id,
            error_code=error_code or "bad_request",
        )
    raise HaluServerError(
        message=message,
        request_id=server_request_id,
        status_code=status,
        upstream_message=detail,
    )


def _parse_detect_response(body: Dict[str, Any]) -> DetectResult:
    try:
        regime_scores = [
            RegimeScore(regime=item["regime"], p=float(item["p"]))
            for item in body.get("regime_scores") or []
        ]
        return DetectResult(
            p_hallucination=float(body["p_hallucination"]),
            flag=bool(body["flag"]),
            top_regime=str(body["top_regime"]),
            regime_scores=regime_scores,
            request_id=str(body.get("request_id", "")),
            detections_billed=int(body.get("detections_billed", 0)),
            mode=str(body.get("mode", "short")),
            latency_ms=int(body.get("latency_ms", 0)),
            model_version=str(body.get("model_version", "")),
            calibrator_version=str(body.get("calibrator_version", "")),
            input_mode_used=str(body.get("input_mode_used", "")),
            task_used=str(body.get("task_used", "")),
            raw=body,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HaluServerError(
            message=f"Malformed detect response: {exc}",
            status_code=200,
        ) from exc


def detect(
    response: str,
    prompt: Optional[str] = None,
    task: str = "multiclass",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> DetectResult:
    """Score an LLM response for hallucination risk.

    Parameters
    ----------
    response : str
        The LLM-generated text to score. Required.
    prompt : str | None
        Optional prompt the LLM saw. Including it selects prompt+response
        mode for higher accuracy on context-dependent claims.
    task : str
        ``"binary"`` or ``"multiclass"``. Defaults to ``"multiclass"`` (full
        regime breakdown).
    api_key : str | None
        API key (``sk_*``). Falls back to ``HALU_API_KEY`` env var.
    base_url : str | None
        Base URL. Falls back to ``HALU_BASE_URL`` env var, then
        ``https://api.komplexai.io``.
    timeout : float | None
        Request timeout in seconds. Falls back to ``HALU_TIMEOUT`` env var,
        then ``30``.

    Returns
    -------
    DetectResult
        Calibrated probability, regime breakdown, and request metadata.

    Raises
    ------
    HaluInputError
        ``response`` is empty or the server rejected the request (400).
    HaluAuthError
        Missing/invalid API key (401). Localhost calls may skip the key.
    HaluQuotaError
        Quota exceeded (402).
    HaluRateLimitError
        Rate limited (429).
    HaluServerError
        Upstream / handler failure (5xx) or malformed response.
    HaluError
        Network or timeout failure (re-raised as :class:`HaluError`).
    """
    if not isinstance(response, str) or response == "":
        raise HaluInputError(
            message="`response` must be a non-empty string.",
            error_code="response_required",
        )
    if task not in ("binary", "multiclass"):
        raise HaluInputError(
            message=f"`task` must be 'binary' or 'multiclass' (got {task!r}).",
            error_code="bad_request",
        )

    resolved_base_url: str = str(_resolve(base_url, "HALU_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    resolved_timeout = float(_resolve(timeout, "HALU_TIMEOUT", DEFAULT_TIMEOUT))
    resolved_key: Optional[str] = _resolve(api_key, "HALU_API_KEY", None)

    if not _is_localhost(resolved_base_url) and not resolved_key:
        raise HaluAuthError(
            message=(
                "No API key found. Set HALU_API_KEY or pass api_key=. "
                "Create a key at https://detector.komplexai.io/account/keys."
            ),
        )

    body: Dict[str, Any] = {"response": response, "task": task}
    if prompt is not None:
        body["prompt"] = prompt

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _user_agent(),
    }
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    url = f"{resolved_base_url}{_DETECT_PATH}"
    try:
        http_response = requests.post(
            url,
            json=body,
            headers=headers,
            timeout=resolved_timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise HaluError(message=f"Request timed out after {resolved_timeout}s.") from exc
    except requests.exceptions.RequestException as exc:
        raise HaluError(message=f"Network error: {exc}") from exc

    request_id = http_response.headers.get("X-Request-Id")
    _raise_for_status(http_response, request_id)

    parsed = _parse_json(http_response.text)
    if parsed is None:
        raise HaluServerError(
            message="Response was not valid JSON.",
            request_id=request_id,
            status_code=http_response.status_code,
        )
    return _parse_detect_response(parsed)


def _user_agent() -> str:
    from . import __version__
    return f"halu-python/{__version__}"
