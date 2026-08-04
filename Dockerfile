FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/claims.db \
    UPLOAD_DIR=/data/uploads \
    MODEL_DIR=/app/backend/models \
    DATA_DIR=/app/backend/data

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/data ./backend/data
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN mkdir -p /data/uploads /app/backend/models && chown -R app:app /app /data
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
