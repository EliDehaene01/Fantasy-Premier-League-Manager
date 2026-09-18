# Build from the REPO ROOT: `docker build -f agents/chips/Dockerfile .`
# Chips' scoring.py imports agents.fixtures.scoring directly (fixture-
# favorability helpers, reused rather than duplicated) - this image needs
# BOTH agent directories, not just its own.
FROM python:3.14-slim

WORKDIR /app

COPY agents/chips/requirements.txt agents/chips/requirements.txt
RUN pip install --no-cache-dir -r agents/chips/requirements.txt

COPY shared/ shared/
COPY ingestion/ ingestion/
COPY agents/fixtures/ agents/fixtures/
COPY agents/chips/ agents/chips/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "agents.chips.service:app", "--host", "0.0.0.0", "--port", "8000"]
