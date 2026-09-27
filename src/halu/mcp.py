"""MCP server for halu — expose the hallucination detector to agents.

Wraps :func:`halu.detect` as a Model Context Protocol (MCP) tool so Claude
Desktop, Claude Code, and any other MCP client can check LLM output for
hallucination risk natively.

Install the MCP extra, then run::

    pip install "halu[mcp]"
    export HALU_API_KEY=sk_...          # from https://detector.komplexai.io/account/keys
    halu-mcp                            # or: python -m halu.mcp

Wire it into an MCP client (e.g. Claude Desktop ``claude_desktop_config.json``)::

    {
      "mcpServers": {
        "halu": {
          "command": "halu-mcp",
          "env": { "HALU_API_KEY": "sk_..." }
        }
      }
    }

The tool never raises to the client — errors are returned as a structured
payload so the agent can reason about quota/auth/transport problems.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import halu


def _build_server() -> "Any":
    """Construct the FastMCP server with the ``detect_hallucination`` tool."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "The MCP server needs the 'mcp' package. Install the extra with:\n"
            '    pip install "halu[mcp]"\n'
        ) from exc

    # Server name = brand ("halu"); the tool name below carries the semantics.
    mcp = FastMCP("halu")

    @mcp.tool()
    def detect_hallucination(
        response: str,
        prompt: Optional[str] = None,
        task: str = "multiclass",
    ) -> Dict[str, Any]:
        """Score an LLM response for hallucination risk.

        Use this to check whether a piece of generated text is accurate and
        trustworthy or likely hallucinated — fabricated, made up, misleading, or
        otherwise unreliable. Reach for it to fact-check an answer, verify agent or
        RAG output, or gauge how much to trust a response before relying on it.

        Args:
            response: The LLM-generated text to score (English, up to 2,048 chars).
            prompt: Optional prompt the model saw. Including it improves accuracy on
                context-dependent claims.
            task: "multiclass" (default; full regime breakdown) or "binary".

        Returns:
            On success: {ok: true, p_hallucination, flag, top_regime, regime_scores,
                         model_version, latency_ms}. ``flag`` is the server's
                         recommended yes/no; ``p_hallucination`` is the calibrated
                         probability (0-1). Regimes: NORMAL, FABRICATED, NEAR_FALSE,
                         CF_AUTH (fake/misattributed citation), FALSE_REFUSAL, Other.
            On failure: {ok: false, error, detail} — e.g. auth, quota, or transport.
        """
        try:
            result = halu.detect(response, prompt=prompt, task=task)
        except halu.HaluError as exc:
            return {
                "ok": False,
                "error": type(exc).__name__,
                "detail": getattr(exc, "message", str(exc)),
            }

        return {
            "ok": True,
            "p_hallucination": result.p_hallucination,
            "flag": result.flag,
            "top_regime": result.top_regime,
            "regime_scores": [rs.to_dict() for rs in result.regime_scores],
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
        }

    return mcp


def main() -> None:
    """Console entry point (``halu-mcp``). Runs the server over stdio."""
    _build_server().run()


if __name__ == "__main__":
    main()
