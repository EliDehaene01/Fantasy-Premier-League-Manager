# Build from the REPO ROOT: `docker build -f ingestion/Dockerfile .`
# A Job, not a Deployment (ARCHITECTURE.md 8) - no HTTP server, just the CLI
# entrypoint already built for exactly this (ingestion/__main__.py's own
# docstring: "the thing a Kubernetes CronJob actually execs"). The k8s Job
# spec supplies the subcommand as args, e.g. `args: ["weekly"]`.
FROM python:3.14-slim

WORKDIR /app

COPY ingestion/requirements.txt ingestion/requirements.txt
RUN pip install --no-cache-dir -r ingestion/requirements.txt

COPY ingestion/ ingestion/

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "ingestion"]
