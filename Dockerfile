FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts
COPY tests/evaluation ./tests/evaluation

RUN python -m pip install --no-cache-dir ".[observability]" \
    && mkdir -p /app/data/chroma /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
