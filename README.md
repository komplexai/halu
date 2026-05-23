# halu

Official Python client library for the [Komplex AI](https://komplexai.io) hallucination-detection API. Score any LLM response for hallucination risk in one HTTP call, plus three small helper functions for the most common usage patterns.

> **API stability — v0.1.0 is the first real release.** The public surface may evolve in 0.2.0 based on real-user feedback. If reproducibility matters for your project, pin a specific version in your requirements.

```bash
pip install halu
```

## Quick start

```python
import halu  # export HALU_API_KEY="sk_..."

result = halu.detect("The Eiffel Tower was built in 1889 by Gustav Eiffel.")
print(result.p_hallucination)  # 0.12
print(result.flag)             # False
print(result.top_regime)       # NORMAL
```

[Get a free API key](https://komplexai.io/account/keys) — no credit card.

## Authentication

`halu.detect()` reads the API key from these sources in priority order:

1. The `api_key=` keyword argument.
2. The `HALU_API_KEY` environment variable.

API keys are issued in the `sk_<base64url>` format from your account dashboard and are sent as an `Authorization: Bearer <key>` header on every call.

Calls against `localhost` / `127.0.0.1` may be made without a key (local detector deployments default to anonymous-allowed). Calls against the production base URL **always** require a key.

```python
import halu

result = halu.detect(
    "...",
    api_key="sk_...",
    base_url="https://komplexai.io",   # default
    timeout=30,                         # seconds
)
```

The `HALU_BASE_URL` and `HALU_TIMEOUT` environment variables override the defaults if no kwarg is supplied.

## The three helpers

Three small wrappers around `detect()` for the most common usage patterns. They are intentionally simple — if your case doesn't fit, call `detect()` directly.

### 1. Gate output — `detect_or_raise`

Raise if the response is flagged at or above `threshold`; return the `DetectResult` otherwise.

```python
from halu import detect_or_raise, HaluHallucinationFlagged

try:
    result = detect_or_raise(llm_response, threshold=0.5)
except HaluHallucinationFlagged as e:
    answer = "I'm not sure — please verify with an expert."
    # e.detection_result has p_hallucination, top_regime, etc.
```

Use when you want to stop bad output from being returned at all.

### 2. Annotate output — `detect_or_warn`

Log a warning if the response is flagged at or above `threshold`; always return the `DetectResult`.

```python
import logging
from halu import detect_or_warn

result = detect_or_warn(llm_response, threshold=0.4)
if result.flag:
    ui.show_banner(f"Verify this — detector p={result.p_hallucination:.2f}")
ui.show(llm_response)
```

Use when you want monitoring/telemetry but no enforcement.

### 3. Auto-regenerate — `regenerate_until_clean`

Call your LLM, run `detect()`, and retry up to `max_retries` times if flagged. Retries can include hallucination-informed feedback so the LLM knows *why* the prior attempt was rejected.

```python
from halu import regenerate_until_clean, HaluRegenerationExhausted

def ask_llm(prompt: str, feedback: str | None = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    if feedback:
        messages.append({"role": "system", "content": feedback})
    return openai_client.chat(messages)

try:
    response, result, history = regenerate_until_clean(
        ask_llm,
        prompt="Who built the Eiffel Tower?",
        max_retries=2,
        acceptance_threshold=0.5,
        feedback_detail="regime",   # "regime" | "binary" | "none"
    )
except HaluRegenerationExhausted as e:
    # e.history is [(response, result), ...] across every attempt
    best = min(e.history, key=lambda pair: pair[1].p_hallucination)
    response = best[0]
```

`feedback_detail="regime"` (default) maps `top_regime` to a specific corrective hint — e.g. `FABRICATED` → "avoid inventing facts that cannot be verified." Use `"binary"` for a generic hint, or `"none"` for blind retry.

`on_exhausted` controls what happens when retries run out without acceptance: `"raise"` (default, raises `HaluRegenerationExhausted`), `"return_best"` (return the attempt with lowest `p_hallucination`), or `"return_last"` (return the last attempt).

### Need something else?

Call `halu.detect()` directly and build your own pattern. The library stays small intentionally.

## What `detect()` returns

A frozen `DetectResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `p_hallucination` | `float` | Calibrated probability (0–1). |
| `flag` | `bool` | True when the server's per-head threshold is exceeded. |
| `top_regime` | `str` | Most likely hallucination regime (`NORMAL`, `FABRICATED`, `CF_AUTH`, `NEAR_FALSE`, etc.). |
| `regime_scores` | `list[RegimeScore]` | All requested regime probabilities. |
| `request_id` | `str` | Server-side request ID for log correlation. |
| `detections_billed` | `int` | Detections consumed by this call (1 at v1). |
| `mode` | `str` | `"short"` (single-pass) or `"long"` (sliding-window). |
| `latency_ms` | `int` | Server-side inference time. |
| `model_version` | `str` | Detector build identifier. |
| `calibrator_version` | `str` | Probability calibrator identifier. |
| `input_mode_used` | `str` | `"pr"` (prompt+response) or `"ro"` (response-only). |
| `task_used` | `str` | `"binary"` or `"multiclass"`. |

Call `result.to_dict()` for JSON-style serialization (e.g. logging).

## Errors

All halu exceptions inherit from `HaluError`. Each carries `.request_id` (when the server returned one) and a typed payload:

| Exception | HTTP | Extra attributes | When raised |
|---|---|---|---|
| `HaluError` | — | — | Base class (transport / timeout). |
| `HaluAuthError` | 401 | — | Missing or invalid API key. |
| `HaluQuotaError` | 402 | `.quota_period`, `.upgrade_url` | Quota exceeded. |
| `HaluRateLimitError` | 429 | `.retry_after` (seconds) | Rate limited. |
| `HaluInputError` | 400 | `.error_code` | Malformed request (empty response, bad task). |
| `HaluServerError` | 5xx | `.status_code`, `.upstream_message` | Upstream/handler failure or malformed response body. |
| `HaluHallucinationFlagged` | — | `.detection_result` | Raised by `detect_or_raise`. |
| `HaluRegenerationExhausted` | — | `.history` | Raised by `regenerate_until_clean`. |

```python
import halu

try:
    result = halu.detect(llm_response)
except halu.HaluQuotaError as e:
    upgrade = e.upgrade_url
except halu.HaluRateLimitError as e:
    time.sleep(e.retry_after or 1)
except halu.HaluAuthError:
    raise SystemExit("Set HALU_API_KEY")
except halu.HaluError as e:
    log.error("halu failed (request_id=%s): %s", e.request_id, e.message)
```

## Wire shape

The library is a thin wrapper around `POST /api/detect`. For raw HTTP usage, see the canonical OpenAPI spec at <https://komplexai.io/api-docs>. A curl-equivalent of the quick-start call:

```bash
curl -X POST https://komplexai.io/api/detect \
  -H "Authorization: Bearer $HALU_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"response": "The Eiffel Tower was built in 1889 by Gustav Eiffel.",
       "task": "multiclass"}'
```

## Post-install smoke test

After `pip install halu` (or installing the wheel directly), you can run an end-to-end smoke test against a local detector deployment to verify the published library actually works on the real HTTP path:

```bash
# Pre-reqs: ulmweb running at localhost:3000 + detector container at localhost:8000.
# The test skips cleanly (exit 0) if either is down.
pytest -m live_local tests/test_e2e_live_local.py
```

Override the targets if your stack lives elsewhere:

```bash
HALU_BASE_URL=http://my-host:3000 \
HALU_DETECTOR_URL=http://my-host:8000 \
  pytest -m live_local tests/test_e2e_live_local.py
```

The smoke verifies: every documented public name is importable; `detect()` flags a known hallucination and clears a clean fact with prompt context; `detect_or_raise` raises `HaluHallucinationFlagged` (with `.detection_result` attached) on flagged; `detect_or_warn` always returns and logs a warning when flagged; `regenerate_until_clean` exhausts cleanly with `raise` / `return_best` / `return_last` (with `.history` populated); and bad inputs raise `HaluInputError` client-side without burning a unit.

## What's new in 0.1.0

- First real release — `detect()` + 3 helper functions.
- Public surface: `detect`, `detect_or_raise`, `detect_or_warn`, `regenerate_until_clean`, `DetectResult`, `RegimeScore`, 8 error classes.
- Typed error hierarchy rooted at `HaluError`.
- Sync only — async deferred to 0.2.0.
- Apache-2.0 license.

## Python versions

Python 3.8+. Tested on 3.8 / 3.9 / 3.10 / 3.11 / 3.12.

## Links

- [komplexai.io](https://komplexai.io) — signup and dashboard
- [`/api-docs`](https://komplexai.io/api-docs) — OpenAPI spec for raw HTTP usage
- [`/guide`](https://komplexai.io/guide) — getting started
- [`/performance`](https://komplexai.io/performance) — benchmark data
- [`/pricing`](https://komplexai.io/pricing) — plans and limits
- [github.com/komplexai/halu](https://github.com/komplexai/halu) — source + [issues](https://github.com/komplexai/halu/issues)

## License

Apache-2.0
