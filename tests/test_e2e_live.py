"""End-to-end tests against a live detector.

These tests are excluded from the default `pytest` run via the `live` marker.
Run with::

    HALU_BASE_URL=http://localhost:3000 pytest -m live

The local detector at `http://localhost:3000/api/detect` currently allows
anonymous requests (no `HALU_API_KEY` required). Against production this
would need a real key.
"""
from __future__ import annotations

import os

import pytest

import halu
from halu import DetectResult

BASE_URL = os.environ.get("HALU_BASE_URL", "http://localhost:3000")

pytestmark = pytest.mark.live


def _assert_well_formed(result: DetectResult) -> None:
    assert isinstance(result, DetectResult)
    assert 0.0 <= result.p_hallucination <= 1.0
    assert isinstance(result.flag, bool)
    assert isinstance(result.top_regime, str) and result.top_regime
    assert len(result.regime_scores) >= 1
    assert all(0.0 <= rs.p <= 1.0 for rs in result.regime_scores)
    assert result.request_id
    assert result.task_used in {"binary", "multiclass"}
    assert result.input_mode_used in {"pr", "ro"}
    assert result.mode in {"short", "long"}
    assert result.latency_ms >= 0


def test_live_clean_fact_no_prompt() -> None:
    r = halu.detect(
        "The Eiffel Tower is in Paris, France, and was completed in 1889.",
        base_url=BASE_URL,
    )
    _assert_well_formed(r)
    print(f"\nclean,ro: p={r.p_hallucination:.3f} flag={r.flag} top={r.top_regime}")


def test_live_clean_fact_with_prompt() -> None:
    r = halu.detect(
        response="The Eiffel Tower is in Paris, France, and was completed in 1889.",
        prompt="Where is the Eiffel Tower?",
        base_url=BASE_URL,
    )
    _assert_well_formed(r)
    assert r.input_mode_used == "pr"
    print(f"\nclean,pr: p={r.p_hallucination:.3f} flag={r.flag} top={r.top_regime}")


def test_live_hallucination_no_prompt() -> None:
    r = halu.detect(
        "The Eiffel Tower is in Berlin and was built in 1823 by Napoleon.",
        base_url=BASE_URL,
    )
    _assert_well_formed(r)
    print(f"\nhall,ro:  p={r.p_hallucination:.3f} flag={r.flag} top={r.top_regime}")


def test_live_hallucination_with_prompt() -> None:
    r = halu.detect(
        response="The Eiffel Tower is in Berlin and was built in 1823 by Napoleon.",
        prompt="Where is the Eiffel Tower?",
        base_url=BASE_URL,
    )
    _assert_well_formed(r)
    assert r.input_mode_used == "pr"
    print(f"\nhall,pr:  p={r.p_hallucination:.3f} flag={r.flag} top={r.top_regime}")


def test_live_task_binary() -> None:
    r = halu.detect(
        "The Eiffel Tower is in Paris.",
        task="binary",
        base_url=BASE_URL,
    )
    _assert_well_formed(r)
    assert r.task_used == "binary"
    print(f"\nbinary:   p={r.p_hallucination:.3f} flag={r.flag} top={r.top_regime}")


def test_live_task_multiclass() -> None:
    r = halu.detect(
        "The Eiffel Tower is in Paris.",
        task="multiclass",
        base_url=BASE_URL,
    )
    _assert_well_formed(r)
    assert r.task_used == "multiclass"
    assert len(r.regime_scores) > 1
    print(f"\nmulti:    p={r.p_hallucination:.3f} flag={r.flag} top={r.top_regime}")
