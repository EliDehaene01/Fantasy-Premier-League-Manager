# Build from the REPO ROOT: `docker build -f agents/stats/Dockerfile .`
# Live serving only needs the trained model artifact, not the archive CSV -
# merged_gws_2025-26.csv (training/backtest input) is deliberately NOT
# copied into this image.
FROM python:3.14-slim

WORKDIR /app

COPY agents/stats/requirements.txt agents/stats/requirements.txt
RUN pip install --no-cache-dir -r agents/stats/requirements.txt

COPY shared/ shared/
COPY ingestion/ ingestion/
COPY features/ features/
COPY agents/stats/ agents/stats/
COPY models/stats_model.pkl models/stats_model.pkl

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "agents.stats.service:app", "--host", "0.0.0.0", "--port", "8000"]
