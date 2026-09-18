# Build from the REPO ROOT: `docker build -f manager/Dockerfile .`
# Manager calls solver.optimizer directly (a Python import, not HTTP - see
# manager/service.py's module docstring) so this image needs solver/ too.
FROM python:3.14-slim

WORKDIR /app

COPY manager/requirements.txt manager/requirements.txt
RUN pip install --no-cache-dir -r manager/requirements.txt

COPY shared/ shared/
COPY solver/ solver/
COPY manager/ manager/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "manager.service:app", "--host", "0.0.0.0", "--port", "8000"]
