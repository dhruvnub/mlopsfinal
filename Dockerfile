# Dockerfile — Experiment 5: Containerize API with Docker
# Push to Azure Container Registry (ACR) via GitHub Actions
#
# Build:  docker build -t placement-api .
# Run:    docker run -p 8000:8000 placement-api
# Docs:   http://localhost:8000/docs

FROM python:3.10-slim

WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only copy what the API needs — not raw data or training code
COPY app.py .
COPY ui.html .
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
