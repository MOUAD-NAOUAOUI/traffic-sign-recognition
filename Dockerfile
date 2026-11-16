# Use official lightweight Python 3.11 Debian image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install minimal OS utilities (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code, model, and assets
COPY src/ ./src/
COPY models/traffic_sign_cnn.keras ./models/
COPY static/ ./static/
COPY sample_signs/ ./sample_signs/
COPY app.py .

# Expose standard application port
EXPOSE 8080

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8080/api/model-info || exit 1

# Start the FastAPI web application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
