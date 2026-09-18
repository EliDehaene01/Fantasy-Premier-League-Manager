# Build from the REPO ROOT: `docker build -f orchestrator/Dockerfile .`
# The graph calls Manager/each specialist over HTTP (orchestrator/callers.py)
# - it does NOT import manager/ or solver/ as Python packages in production,
# only manager.schemas' request/response shapes for building requests, so
# those two thin schema modules are copied in without the rest of either
# service.
FROM python:3.14-slim

WORKDIR /app

COPY orchestrator/requirements.txt orchestrator/requirements.txt
RUN pip install --no-cache-dir -r orchestrator/requirements.txt

COPY shared/ shared/
COPY solver/schemas.py solver/schemas.py
COPY solver/__init__.py solver/__init__.py
COPY manager/schemas.py manager/schemas.py
COPY manager/__init__.py manager/__init__.py
COPY orchestrator/ orchestrator/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "orchestrator.service:app", "--host", "0.0.0.0", "--port", "8000"]
