# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including audio tools and Rhubarb dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    unzip \
    libasound2-dev \
    libpulse-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create bin directory and download Rhubarb Lip Sync
RUN mkdir -p bin && \
    cd bin && \
    wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip && \
    unzip rhubarb-lip-sync-1.13.0-linux.zip && \
    chmod +x Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb && \
    rm rhubarb-lip-sync-1.13.0-linux.zip

# Copy application code
COPY . .

# Create audios directory
RUN mkdir -p audios

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 