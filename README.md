# halu

Hallucination detection for LLM responses.

```bash
pip install halu
```

## Quick start

No API key required on the free tier.

```python
import halu

result = halu.detect("The Eiffel Tower was built in 1889 by Gustav Eiffel.")
print(result.verdict)          # HIGH
print(result.p_hallucination)  # 0.73
print(result.top_regime)       # FABRICATED_CLAIM
```

## With an API key

```python
import halu

client = halu.Client(api_key="YOUR_KEY")
result = client.detect(response=llm_response, prompt=my_prompt)

if result.verdict == "HIGH":
    # raise, retry, flag — your call
    pass
```

## What it returns

| Field | Type | Description |
|---|---|---|
| `p_hallucination` | float 0–1 | Calibrated probability of hallucination |
| `verdict` | str | `LOW` / `MODERATE` / `HIGH` |
| `top_regime` | str | Most likely hallucination pattern |
| `regime_scores` | list | Top-k regimes with scores |
| `latency_ms` | int | Server-side inference time |

## Links

- [Docs & API reference](https://komplexai.io/guide)
- [Pricing & rate limits](https://komplexai.io/pricing)
- [komplexai.io](https://komplexai.io)

## License

MIT
