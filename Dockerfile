# ============================================================
# Base Image
# ============================================================
FROM python:3.12-slim

# Prevent Python from writing pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ============================================================
# Install system dependencies (optional but safe)
# ============================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Copy dependencies & install them
# ============================================================
COPY requirements.txt .
RUN PIP_DEFAULT_TIMEOUT=1000 pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy application code
# ============================================================
COPY ./app ./app

# ============================================================
# Expose FastAPI port & define startup command
# ============================================================
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

