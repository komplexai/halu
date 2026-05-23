"""Exception hierarchy for halu.

All halu-raised exceptions inherit from :class:`HaluError`. Server-side
errors carry the originating ``request_id`` when available so callers can
correlate logs across hops.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ._types import DetectResult


class HaluError(Exception):
    """Base class for every exception raised by the halu library."""

    def __init__(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id

    def __str__(self) -> str:
        if self.request_id:
            return f"{self.message} (request_id={self.request_id})"
        return self.message


class HaluAuthError(HaluError):
    """Authentication failed: missing API key or 401 from the server."""


class HaluQuotaError(HaluError):
    """Quota exceeded (HTTP 402). ``quota_period`` is the billing-period anchor (e.g. ``"2026-05"``)."""

    def __init__(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
        quota_period: Optional[str] = None,
        upgrade_url: Optional[str] = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.quota_period = quota_period
        self.upgrade_url = upgrade_url


class HaluRateLimitError(HaluError):
    """Rate limited (HTTP 429). ``retry_after`` is the integer seconds from the ``Retry-After`` header, if present."""

    def __init__(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.retry_after = retry_after


class HaluInputError(HaluError):
    """The request was malformed (HTTP 400). ``error_code`` is the machine-readable error class (e.g. ``response_required``)."""

    def __init__(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.error_code = error_code


class HaluServerError(HaluError):
    """Server-side failure (HTTP 5xx). Carries the raw ``status_code`` and the upstream message if any."""

    def __init__(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
        status_code: Optional[int] = None,
        upstream_message: Optional[str] = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.status_code = status_code
        self.upstream_message = upstream_message


class HaluHallucinationFlagged(HaluError):
    """Raised by :func:`halu.detect_or_raise` when a response is flagged at the threshold.

    The full :class:`DetectResult` is attached at ``.detection_result`` so callers
    can inspect ``p_hallucination``, ``top_regime``, etc. without re-calling the API.
    """

    def __init__(
        self,
        message: str,
        *,
        detection_result: "DetectResult",
    ) -> None:
        super().__init__(message, request_id=detection_result.request_id)
        self.detection_result = detection_result


class HaluRegenerationExhausted(HaluError):
    """Raised by :func:`halu.regenerate_until_clean` when retries are exhausted without acceptance.

    ``.history`` is the full list of ``(response, detection_result)`` attempts in order,
    so callers can inspect every regeneration to choose the best one or surface the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        history: "List[Tuple[str, DetectResult]]",
    ) -> None:
        last_request_id = history[-1][1].request_id if history else None
        super().__init__(message, request_id=last_request_id)
        self.history = history
