# DeepTrace - CPU-only container (Render / Cloud Run / Hugging Face Spaces / Fly)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCV_LOG_LEVEL=ERROR \
    DEEPTRACE_DATA_DIR=/data \
    DEEPTRACE_WORK_DIR=/data/runs

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /data/runs && chmod +x docker-entrypoint.sh

# 7860 is what Hugging Face Spaces expects; override with $PORT elsewhere
ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,sys,urllib.request;sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health', timeout=4).status==200 else 1)"

CMD ["./docker-entrypoint.sh"]
