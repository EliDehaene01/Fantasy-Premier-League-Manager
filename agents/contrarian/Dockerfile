# Build from the REPO ROOT: `docker build -f agents/contrarian/Dockerfile .`
FROM python:3.14-slim

WORKDIR /app

COPY agents/contrarian/requirements.txt agents/contrarian/requirements.txt
RUN pip install --no-cache-dir -r agents/contrarian/requirements.txt

COPY shared/ shared/
COPY ingestion/ ingestion/
COPY agents/contrarian/ agents/contrarian/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "agents.contrarian.service:app", "--host", "0.0.0.0", "--port", "8000"]
