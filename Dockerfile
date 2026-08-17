# Step 1: Base image set to Python 3.13 slim
FROM python:3.13-slim

# Step 2: Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Step 3: Set working directory
WORKDIR /app

# Step 4: Install system dependencies (uncomment if using packages like psycopg2 or Pillow)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libpq-dev \
#  && rm -rf /var/lib/apt/lists/*

# Step 5: Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Step 7: Copy application source code
COPY . .

# Step 8: Expose default Render port
EXPOSE 10000

# Step 9: Start Gunicorn bound dynamically to Render's $PORT
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]