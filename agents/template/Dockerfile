# Build from the REPO ROOT: `docker build -f agents/template/Dockerfile .`
FROM python:3.14-slim

WORKDIR /app

COPY agents/template/requirements.txt agents/template/requirements.txt
RUN pip install --no-cache-dir -r agents/template/requirements.txt

COPY shared/ shared/
COPY ingestion/ ingestion/
COPY agents/template/ agents/template/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "agents.template.service:app", "--host", "0.0.0.0", "--port", "8000"]
