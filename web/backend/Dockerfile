# Granite Studios web API — build context is the REPO ROOT, not this folder,
# because the service imports the engines from src/.
#   docker build -f web/backend/Dockerfile -t granite-web .
FROM python:3.11-slim

# libGL/libglib: OpenCV runtime. ffmpeg: system build preferred over the
# imageio-ffmpeg fallback. libgomp1: onnxruntime threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY web/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY web/backend/ ./web/backend/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GRANITE_STORAGE_ROOT=/tmp/granite-web \
    NUMBA_CACHE_DIR=/tmp \
    U2NET_HOME=/tmp/u2net

# Pre-download the background-removal model at build time so the first user
# request isn't blocked on a ~170MB download.
RUN python -c "from rembg import new_session; new_session()" || \
    echo "Model pre-fetch skipped; it will download on first use."

WORKDIR /app/web/backend
EXPOSE 8000

# One worker: video encoding is CPU-bound, and extra workers would multiply
# memory against the same cores. Long timeout for heavy video jobs.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]
