# Use Python slim image (small, fast)
FROM python:3.11-slim

# Patch OS packages and bundled wheel (CVE-2026-24049 in python:3.11-slim)
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade "pip>=26.0" "wheel>=0.46.2"

# Set working directory
WORKDIR /app

# Copy requirements first (docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/raw models logs

# Set entrypoint
CMD ["python", "main.py"]