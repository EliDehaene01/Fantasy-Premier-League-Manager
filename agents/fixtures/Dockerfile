# Build from the REPO ROOT, not this directory: `docker build -f agents/fixtures/Dockerfile .`
# This agent imports shared/ (contract + Foundry client) and ingestion/db.py
# (Postgres connection) across package boundaries - a monorepo, so the build
# context has to be the repo root for those COPYs to reach.
FROM python:3.14-slim

WORKDIR /app

COPY agents/fixtures/requirements.txt agents/fixtures/requirements.txt
RUN pip install --no-cache-dir -r agents/fixtures/requirements.txt

COPY shared/ shared/
COPY ingestion/ ingestion/
COPY agents/fixtures/ agents/fixtures/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "agents.fixtures.service:app", "--host", "0.0.0.0", "--port", "8000"]
