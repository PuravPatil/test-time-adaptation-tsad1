# FastAPI backend for the demo. Build from the project root (docker-compose.yml does):
#   docker compose build api && docker compose up api       -> http://localhost:8000/docs
# CPU-only, like the main image: jobs train tiny MLPs and adapt one window at a time.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TSAD_ROOT=/app

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY demo/api/requirements.txt ./demo/api/requirements.txt
RUN pip install -r requirements.txt -r demo/api/requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY configs ./configs
COPY demo/api ./demo/api

# data/ and results/ are mounted as volumes (see docker-compose.yml).
EXPOSE 8000
CMD ["uvicorn", "tsad_api.main:create_app", "--factory", "--app-dir", "demo/api", "--host", "0.0.0.0", "--port", "8000"]
