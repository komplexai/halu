# halu examples

Runnable integration examples for the [`halu`](https://pypi.org/project/halu/)
hallucination-detection SDK. Each is a **self-contained example**, not a
sub-library — copy the pieces you need.

First:

```bash
pip install halu
export HALU_API_KEY=sk_...   # from https://detector.komplexai.io/account/keys
```

## `langchain_integration.py`

Add hallucination detection to any LangChain chain. Three patterns:

| Pattern | What it does | When to use |
|---------|--------------|-------------|
| **annotate** | Pass the score through alongside the answer | Monitoring / UI badge; never blocks the user |
| **gate** | Raise / replace the answer when flagged | Stop bad output from reaching the user |
| **retry** | Regenerate with regime-specific feedback until clean | Recover automatically |

Uses only `langchain_core`, so it works with any chat model.

```bash
pip install langchain-core
python examples/langchain_integration.py
```

## `mcp_server.py`

Expose the detector as a [Model Context Protocol](https://modelcontextprotocol.io)
tool so Claude Desktop, Claude Code, and other MCP clients can call it.

```bash
pip install mcp
python examples/mcp_server.py
```

Then add to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "halu": {
      "command": "python",
      "args": ["/absolute/path/to/examples/mcp_server.py"],
      "env": { "HALU_API_KEY": "sk_..." }
    }
  }
}
```

Once connected, the client can call the `detect_hallucination` tool autonomously
when a task calls for checking whether generated text is trustworthy. For
*reliable* checking (every generation), call it deterministically from an agent
loop — see the `gate` / `retry` patterns in the LangChain example.

## More

- Docs & API reference: <https://detector.komplexai.io/guide>
- OpenAPI spec (for raw HTTP / other languages): <https://detector.komplexai.io/api-docs>
- Issues / requests: <https://github.com/komplexai/halu/issues>
