FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY rescue_env/ ./rescue_env/
COPY baselines/ ./baselines/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY openenv.yaml ./openenv.yaml
COPY app.py ./app.py
COPY inference.py ./inference.py
COPY README.md ./README.md

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
