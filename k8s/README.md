# Kubernetes manifests

Local Docker Desktop Kubernetes only - not a cloud cluster.

## Deploy

```bash
# 1. Build every image locally (Docker Desktop's k8s shares the docker daemon's
#    image cache, so imagePullPolicy: Never works without a registry).
docker build -t fpl-agents/solver:local -f solver/Dockerfile .
docker build -t fpl-agents/manager:local -f manager/Dockerfile .
docker build -t fpl-agents/orchestrator:local -f orchestrator/Dockerfile .
docker build -t fpl-agents/stats:local -f agents/stats/Dockerfile .
docker build -t fpl-agents/fixtures:local -f agents/fixtures/Dockerfile .
docker build -t fpl-agents/news:local -f agents/news/Dockerfile .
docker build -t fpl-agents/contrarian:local -f agents/contrarian/Dockerfile .
docker build -t fpl-agents/template:local -f agents/template/Dockerfile .
docker build -t fpl-agents/chips:local -f agents/chips/Dockerfile .
docker build -t fpl-agents/ingestion:local -f ingestion/Dockerfile .

# 2. Namespace, config, secrets first.
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
cp k8s/secret.example.yaml k8s/secret.yaml   # then fill in real values - never commit secret.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/postgres-external.yaml

# 3. Everything else.
kubectl apply -f k8s/solver.yaml -f k8s/stats.yaml -f k8s/fixtures.yaml -f k8s/news.yaml \
  -f k8s/contrarian.yaml -f k8s/template.yaml -f k8s/chips.yaml -f k8s/manager.yaml \
  -f k8s/orchestrator.yaml
kubectl apply -f k8s/deadline-checker-cronjob.yaml
```

## Verify

```bash
kubectl get pods -n fpl-agents
kubectl port-forward -n fpl-agents svc/orchestrator 8000:8000
curl http://localhost:8000/health
```

## Postgres: external, not in-cluster

Deliberate - see postgres-external.yaml's own comment for the full reasoning
(TODO.md Phase 5 explicitly calls for a documented decision here, not a
default). Short version: the local cluster exists to demonstrate
orchestration skill, not to be a durable data store, and the existing
`fpl-postgres` container is already the established setup.

