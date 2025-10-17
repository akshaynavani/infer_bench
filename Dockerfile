FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY workloads ./workloads

EXPOSE 8000 9646 8089
CMD ["infer-bench", "serve", "--host", "0.0.0.0", "--port", "8000"]
