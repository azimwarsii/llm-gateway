.PHONY: dev test lint up-small up-large down bench judge audit chaos k8s

dev:            ## run gateway locally (no docker) against whatever backends config points at
	GATEWAY_CONFIG=config/gateway.yaml uvicorn gateway.main:app --port 8080 --reload

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

up-small:       ## gateway + redis + grafana + Qwen2.5-3B on vLLM
	docker compose --profile small up -d --build

up-large:       ## gateway + redis + grafana + Llama-3.1-8B-AWQ on vLLM
	docker compose --profile large up -d --build

down:
	docker compose --profile small --profile large down

bench:          ## 200 prompts x (small, remote, auto)
	python bench/run_bench.py --modes small remote auto --concurrency 2

judge:          ## grade the latest results with the remote tier
	python bench/judge.py $$(ls -t bench/results/results-*.jsonl | head -1)

audit:          ## router tier vs. difficulty label
	python bench/audit_router.py $$(ls -t bench/results/results-*.jsonl | head -1)

chaos:          ## kill/restart vllm-small while sending traffic
	python bench/fault_injection.py --requests 100 --kill-every 15

k8s:            ## deploy to a local kind cluster
	kind create cluster --name gw || true
	docker build -t llm-gateway:dev .
	kind load docker-image llm-gateway:dev --name gw
	kubectl apply -k k8s/
