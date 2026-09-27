"""halu — Python client for the Komplex AI hallucination-detection API.

Quick start::

    import halu
    result = halu.detect("The Eiffel Tower was built in 1889 by Gustav Eiffel.")
    print(result.p_hallucination, result.top_regime)

Three helper functions cover the most common usage patterns:

- :func:`detect_or_raise` — gate output (raise when flagged)
- :func:`detect_or_warn`  — annotate output (log a warning when flagged)
- :func:`regenerate_until_clean` — recover via LLM-informed retry

For anything else, call :func:`detect` directly and build your own pattern.

See https://github.com/komplexai/halu for full documentation.
"""
from __future__ import annotations

__version__ = "0.1.3"

from ._client import detect
from ._errors import (
    HaluAuthError,
    HaluError,
    HaluHallucinationFlagged,
    HaluInputError,
    HaluQuotaError,
    HaluRateLimitError,
    HaluRegenerationExhausted,
    HaluServerError,
)
from ._helpers import detect_or_raise, detect_or_warn, regenerate_until_clean
from ._types import DetectResult, RegimeScore

__all__ = [
    "__version__",
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
]
