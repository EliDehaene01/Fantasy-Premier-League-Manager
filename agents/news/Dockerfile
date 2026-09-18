# Build from the REPO ROOT: `docker build -f agents/news/Dockerfile .`
# severity_classifier.py imports agents.stats.data for its REPO_ROOT
# constant (see agents/news/requirements.txt's pandas note) - this image
# needs agents/stats/ present too, though only data.py is actually used.
# playwright's browser binaries are installed separately (see the RUN step
# below) - the pip package alone doesn't bundle them.
FROM python:3.14-slim

WORKDIR /app

COPY agents/news/requirements.txt agents/news/requirements.txt
RUN pip install --no-cache-dir -r agents/news/requirements.txt \
    && playwright install --with-deps chromium

COPY shared/ shared/
COPY ingestion/ ingestion/
COPY agents/stats/ agents/stats/
COPY agents/news/ agents/news/
COPY models/severity_classifier.joblib models/severity_classifier.joblib

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "agents.news.service:app", "--host", "0.0.0.0", "--port", "8000"]
