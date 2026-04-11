FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Keep package paths under /app so local Docker Compose and Cloud Run can both
# import the selected serving entrypoint and the canonical top-level feature_repo package.
COPY serving /app/serving
COPY feature_repo /app/feature_repo

EXPOSE 8000
CMD ["sh", "-c", "uvicorn serving.realtime_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
