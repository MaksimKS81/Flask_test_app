# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install system deps (if you use pandas / authlib etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev libpq-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for cache)
COPY requirements.txt .

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container working dir
COPY . /app/

# Expose port (Cloud Run sets $PORT automatically)
EXPOSE 8080

# Default to use gunicorn with Flask app
CMD ["gunicorn", "--preload", "--timeout", "120", "-b", ":8080", "app:app"]

