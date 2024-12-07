FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Set longer timeout and retry mechanism for pip
RUN pip install --no-cache-dir --timeout 200 --retries 3 torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout 200 --retries 3 -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]