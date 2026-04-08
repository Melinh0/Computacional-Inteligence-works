FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

COPY pyproject.toml .

RUN uv sync

COPY src ./src

CMD ["uv", "run", "python", "src/experiment.py"]