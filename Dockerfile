# CPU-only image: the models are tiny MLPs and test-time adaptation updates one
# window at a time (batch size 1), where a GPU is slower than the CPU. CPU also
# keeps numerics identical across machines (see configs/base.yaml: device: cpu).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

# Dependencies first so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY results/paper_reference.json ./results/paper_reference.json

# data/ and results/ are mounted as volumes (see docker-compose.yml), not baked in.
CMD ["python", "scripts/run.py", "--config", "configs/experiments/smd_quick.yaml"]
