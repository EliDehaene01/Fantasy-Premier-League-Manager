# Build from the REPO ROOT: `docker build -f solver/Dockerfile .`
FROM python:3.14-slim

WORKDIR /app

COPY solver/requirements.txt solver/requirements.txt
RUN pip install --no-cache-dir -r solver/requirements.txt

COPY shared/ shared/
COPY solver/ solver/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "solver.service:app", "--host", "0.0.0.0", "--port", "8000"]
