"""Result dataclasses matching the ``/api/detect`` wire shape.

Frozen dataclasses for immutability. Field names mirror the canonical
OpenAPI schema in ``docs/api/openapi.yaml``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class RegimeScore:
    """One row of ``regime_scores`` — a regime code with its calibrated probability."""

    regime: str
    p: float

    def to_dict(self) -> Dict[str, Any]:
        return {"regime": self.regime, "p": self.p}


@dataclass(frozen=True)
class DetectResult:
    """Successful response from ``POST /api/detect``.

    Attribute names match the wire schema. Use :meth:`to_dict` for JSON-style
    serialization (e.g. logging, audit trails).
    """

    p_hallucination: float
    flag: bool
    top_regime: str
    regime_scores: List[RegimeScore]
    request_id: str
    detections_billed: int
    mode: str
    latency_ms: int
    model_version: str
    calibrator_version: str
    input_mode_used: str
    task_used: str
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["regime_scores"] = [rs.to_dict() for rs in self.regime_scores]
        d.pop("raw", None)
        return d
