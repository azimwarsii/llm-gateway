from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("gateway_requests_total", "Requests by backend, tier and outcome", ["backend", "tier", "outcome"])
LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "End-to-end latency",
    ["backend", "tier"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
TTFT = Histogram(
    "gateway_ttft_seconds", "Time to first streamed token", ["backend"], buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4)
)
TOKENS = Counter("gateway_tokens_total", "Tokens by backend and direction", ["backend", "direction"])
COST = Counter("gateway_cost_usd_total", "Accounted cost in USD", ["backend", "tenant"])
CACHE = Counter("gateway_cache_total", "Cache hits and misses", ["result"])
FALLBACKS = Counter("gateway_fallbacks_total", "Fallbacks from one tier to another", ["from_tier", "to_tier"])
CIRCUIT = Gauge("gateway_circuit_open", "1 if the backend circuit is open", ["backend"])
BACKEND_UP = Gauge("gateway_backend_up", "1 if the backend health check passed", ["backend"])
ROUTE_SCORE = Histogram(
    "gateway_route_score", "Router complexity score", buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)
