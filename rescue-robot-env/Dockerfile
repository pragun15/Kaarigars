FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY rescue_env/ ./rescue_env/
COPY baselines/ ./baselines/
COPY configs/ ./configs/
COPY openenv.yaml ./openenv.yaml

CMD ["python", "-m", "baselines.run_baseline", "--mode", "mock", "--difficulty", "easy", "--episodes", "5", "--seed", "42"]
