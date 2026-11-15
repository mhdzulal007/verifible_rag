FROM python:3.12-slim

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OCR & PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and benchmark files
COPY src/ ./src/
COPY data/ ./data/
COPY api.py .
COPY app.py .
COPY run_evaluation.py .

EXPOSE 8000 8501

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
