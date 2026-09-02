# Kubernetes (kind)

These manifests run the gateway + Redis on a local kind cluster. vLLM stays on the host
(Docker Compose) because GPU passthrough into kind is fiddly; the gateway reaches it via
`host.docker.internal`. This is enough to exercise Deployments, Services, ConfigMaps,
probes and HPA behaviour locally.

```bash
kind create cluster --name gw
docker build -t llm-gateway:dev .
kind load docker-image llm-gateway:dev --name gw
kubectl apply -k k8s/
kubectl -n llm-gateway port-forward svc/gateway 8080:8080
curl localhost:8080/health
```
