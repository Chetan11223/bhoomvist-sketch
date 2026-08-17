# Step 1: Base image
FROM python:3.13-slim

# Step 2: Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Step 3: Set working directory
WORKDIR /app

# Step 4: Install required system packages for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Step 5: Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Step 7: Copy application source code
COPY --chown=appuser:appuser . .

# Step 8: Expose default port
EXPOSE 10000



CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]