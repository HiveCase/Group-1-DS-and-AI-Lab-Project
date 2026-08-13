FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/claims.db \
    UPLOAD_DIR=/data/uploads \
    MODEL_DIR=/app/models \
    DATA_DIR=/data

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

# Native deps for ultralytics/opencv (image decoding) and pdfplumber's C libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
# Empty by default (secure, and what CI uses on GitHub's clean network).
# Behind a corporate TLS-inspecting proxy that breaks pip's cert
# verification, build locally with e.g.:
#   docker build --build-arg PIP_EXTRA_ARGS="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org" -t claims-portal .
ARG PIP_EXTRA_ARGS=""
# requirements.txt doesn't list torch directly -- it's pulled in
# transitively by ultralytics and sentence-transformers, and PyPI's default
# torch wheel bundles the full CUDA/cuDNN/cuBLAS stack (~1.5-2GB) that this
# app never uses (CPU-only inference on a small YOLO model, no GPU). Installing
# the CPU-only build from PyTorch's own index first means pip's resolver
# treats torch as already satisfied when it hits ultralytics/
# sentence-transformers below, instead of reaching for the CUDA build.
RUN pip install --no-cache-dir ${PIP_EXTRA_ARGS} torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ${PIP_EXTRA_ARGS} -r requirements.txt

COPY backend/app ./app
COPY backend/models ./models
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN mkdir -p /data/uploads && chown -R app:app /app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
