"""Example: run the halu MCP server.

The server now ships *inside* the SDK, so the one-command way to run it is::

    pip install "halu[mcp]"
    export HALU_API_KEY=sk_...
    halu-mcp                     # or: python -m halu.mcp

This example file just invokes that same packaged server, so you can run it
directly from a checkout::

    python examples/mcp_server.py

See ``halu/mcp.py`` for the implementation and the Claude Desktop config block.
"""
from halu.mcp import main

if __name__ == "__main__":
    main()
