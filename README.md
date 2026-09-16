# llm-gateway

An OpenAI-compatible gateway that sits in front of self-hosted open models and a hosted API,
and decides per request which one to use.

- **Routes by complexity**: cheap local model for easy requests, bigger local model for medium,
  hosted API for hard ones. Every decision is explainable and returned in the response.
- **Survives failures**: retries with jittered backoff, a circuit breaker per backend, health
  checks, and automatic fallback up the tier list. A dead backend costs a few ms, not a request.
- **Controls spend**: per-tenant API keys, daily token budgets, per-minute rate limits,
  a Redis response cache, and accounted cost per request.
- **Observable**: Prometheus metrics (latency, TTFT, tokens, cost, cache, fallbacks, circuit
  state) with a provisioned Grafana dashboard.
- **Measured**: a 200-prompt benchmark, a rubric judge, a router audit and a chaos test.

Built and tested on a single RTX 3070 (8 GB). Everything runs locally with Docker Compose;
Kubernetes manifests are included and tested on kind.

<!-- TODO: replace with a real GIF of a request going through the gateway + the Grafana dashboard -->
![demo](docs/demo.gif)

## Quick start

```bash
git clone https://github.com/AzimWarsii/llm-gateway && cd llm-gateway
cp .env.example .env          # add GROQ_API_KEY (free tier) for the hosted fallback tier

# Start gateway + redis + prometheus + grafana + Qwen2.5-3B on vLLM (~2.5 GB VRAM)
docker compose --profile small up -d --build

curl localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer dev-key' \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

The response is standard OpenAI JSON plus a `gateway` block:

```json
"gateway": {
  "backend": "qwen-3b", "tier": "small", "requested_tier": "small",
  "route_score": 0.016, "route_reason": "complexity score",
  "latency_ms": 412.7, "cost_usd": 0.0000041, "cache": "miss",
  "attempts": [{"backend": "qwen-3b", "tier": "small", "outcome": "ok", "latency_ms": 412.7}]
}
```

Any OpenAI SDK works unchanged: set `base_url="http://localhost:8080/v1"` and `api_key="dev-key"`.
Use `model="auto"` for routing, or pin a tier with `model="small" | "medium" | "remote"`.

- Grafana: http://localhost:3000 (dashboard "LLM Gateway")
- Prometheus: http://localhost:9090
- Gateway status: `curl -H 'X-Admin-Key: change-me' localhost:8080/admin/status`

### Running the 8B model

The 3070 cannot hold both models at once, so they are behind Compose profiles:

```bash
docker compose --profile small down
docker compose --profile large up -d      # Llama-3.1-8B-Instruct AWQ, ~5.5 GB VRAM
```

The gateway notices which one is up and routes around the other. In the benchmark the
`medium` tier was measured in a separate pass with the `large` profile running.

## Architecture

```
client ──► FastAPI gateway ──► router (complexity score → tier)
                │                    │
                │                    ├─ small   vLLM  Qwen2.5-3B (AWQ)      local GPU
                │                    ├─ medium  vLLM  Llama-3.1-8B (AWQ)    local GPU
                │                    └─ remote  Groq  Llama-3.3-70B         hosted
                │
                ├─ per-backend circuit breaker + retries + health checks + fallback ↑
                ├─ Redis: response cache, tenant budgets, rate limits
                └─ Prometheus /metrics  ──►  Grafana
```

**Router.** `gateway/router.py` computes a score in [0, 1] from cheap features: prompt
length, turn count, reasoning/code/math hints, tool use, requested output length, and a
penalty for obviously simple asks. Thresholds were calibrated against the benchmark's
difficulty labels (see *Router audit*). The score and features are returned in every
response so routing is auditable. Swapping in a trained classifier means replacing one function.

**Failure handling.** Timeouts, 5xx and 429 are retryable (jittered exponential backoff,
`max_retries` per backend); other 4xx are fatal for that backend. Three consecutive failures
open the backend's circuit for 30 s, after which one probe request is allowed through.
An unhealthy or open backend is skipped in a few microseconds and the request falls back to
the next tier. For streaming, fallback happens only before the first byte is sent.

**Token management.** Each tenant has a daily token budget and requests-per-minute limit,
enforced with atomic Redis counters shared across replicas. Cost is accounted per request from
per-backend prices (local models use an electricity + amortisation estimate; hosted uses list
price so comparisons are meaningful even on a free tier). `GET /v1/usage` shows remaining budget.

## Benchmark

`bench/prompts.jsonl` has 200 prompts across 8 categories (factual, rewrite, extraction,
small code, reasoning, large code, analysis, long context), each with a difficulty label.

```bash
make bench     # runs small-only, remote-only and auto over all 200 prompts
make judge     # scores every answer 1-5 with the remote tier as judge
make audit     # router tier vs. difficulty label
```

<!-- TODO: replace the table below with your measured numbers from bench/results/summary-*.json
     and the judge output. The rows are the shape of the result, not real measurements. -->

| mode | success | p50 latency | p95 latency | cost / 1K req | mean quality (1-5) | quality vs remote |
|------|---------|-------------|-------------|---------------|--------------------|-------------------|
| small (3B)   | | | | | | |
| medium (8B)  | | | | | | |
| remote (70B) | | | | | | 100% |
| **auto**     | | | | | | |

Notes on method: temperature 0, `max_tokens` 512, cache bypassed (`X-Cache: bypass`).
The judge is the remote tier, which biases quality scores toward that model family; use a
different provider as judge (`bench/judge.py --judge-url ...`) for a fairer comparison.

### Router audit

Cross-tab of the router's chosen tier against the human difficulty label on the 200 prompts
(after calibration; numbers from a run against mock backends, routing does not depend on the model):

```
difficulty ->  small  medium  remote
easy       ->     90       0       0
medium     ->     11      54       0
hard       ->      7      25      13
```

### Quantization on 8 GB

<!-- TODO: fill from your runs. Suggested: Qwen2.5-3B fp16 vs AWQ; Llama-3.1-8B AWQ. -->

| model | precision | VRAM | tok/s (1 seq) | tok/s (8 seq) | p95 TTFT | mean quality |
|-------|-----------|------|---------------|---------------|----------|--------------|
| Qwen2.5-3B | fp16 | | | | | |
| Qwen2.5-3B | AWQ int4 | | | | | |
| Llama-3.1-8B | AWQ int4 | | | | | |

### Chaos test

`make chaos` kills and restarts the small vLLM container every 15 s while sending 100
requests through the gateway, and reports the success rate and how many requests fell back.

<!-- TODO: paste the JSON output here. -->

## Development

```bash
pip install -e ".[dev,bench]"
make test      # 17 tests against mocked backends (no GPU needed)
make lint
make dev       # gateway on :8080 with --reload
```

Tests cover routing, cache hit/bypass, retry-then-fallback, circuit opening, all-backends-down,
fatal 4xx handling, tenant auth, tier restrictions, rate limit and budget, and streaming
passthrough with pre-first-byte fallback.

## Kubernetes

`k8s/` has a kustomize bundle (namespace, Redis, gateway Deployment with probes and an HPA,
ConfigMap from `config/gateway.yaml`). `make k8s` builds the image, loads it into a kind
cluster and applies it. vLLM stays on the host. See `k8s/README.md`.

## Limitations

- One GPU, so only one local model is resident at a time; the medium tier is measured in a separate pass.
- The router is a hand-tuned heuristic. It is calibrated to this prompt set and will need retuning for other traffic; a trained classifier is the next step.
- Semantic caching (embedding similarity) is not implemented; the cache is exact-match on normalised requests at temperature 0.
- No auth beyond static API keys; no request logging to durable storage.
- Streaming responses estimate token usage rather than reading it from the backend.

## License

MIT
