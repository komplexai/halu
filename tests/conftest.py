"""Shared pytest fixtures for the halu test suite."""
from __future__ import annotations

from typing import Any, Dict

import pytest


@pytest.fixture
def detect_response_ok() -> Dict[str, Any]:
    """A canonical, well-formed `POST /api/detect` 200 body."""
    return {
        "p_hallucination": 0.12,
        "flag": False,
        "top_regime": "NORMAL",
        "regime_scores": [
            {"regime": "NORMAL", "p": 0.72},
            {"regime": "FABRICATED", "p": 0.11},
            {"regime": "NEAR_FALSE", "p": 0.07},
            {"regime": "UNDERSPECIFIED", "p": 0.04},
            {"regime": "SELF_CONTR", "p": 0.03},
            {"regime": "CF_AUTH", "p": 0.02},
            {"regime": "FALSE_REFUSAL", "p": 0.01},
        ],
        "request_id": "req_test_clean_0001",
        "detections_billed": 1,
        "mode": "short",
        "latency_ms": 412,
        "model_version": "nl-v1",
        "calibrator_version": "platt-v3",
        "input_mode_used": "ro",
        "task_used": "multiclass",
    }


@pytest.fixture
def detect_response_flagged() -> Dict[str, Any]:
    """A canonical 200 body with `flag=True`."""
    return {
        "p_hallucination": 0.87,
        "flag": True,
        "top_regime": "FABRICATED",
        "regime_scores": [
            {"regime": "FABRICATED", "p": 0.61},
            {"regime": "CF_AUTH", "p": 0.18},
            {"regime": "NEAR_FALSE", "p": 0.09},
            {"regime": "NORMAL", "p": 0.06},
            {"regime": "SELF_CONTR", "p": 0.03},
            {"regime": "UNDERSPECIFIED", "p": 0.02},
            {"regime": "FALSE_REFUSAL", "p": 0.01},
        ],
        "request_id": "req_test_flagged_0002",
        "detections_billed": 1,
        "mode": "short",
        "latency_ms": 388,
        "model_version": "nl-v1",
        "calibrator_version": "platt-v3",
        "input_mode_used": "pr",
        "task_used": "multiclass",
    }


@pytest.fixture(autouse=True)
def _clear_halu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests start from a clean env — no HALU_* leaking from the shell."""
    for var in ("HALU_API_KEY", "HALU_BASE_URL", "HALU_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
