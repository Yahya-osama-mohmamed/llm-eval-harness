FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dev]"

COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Default to proving the build is sound rather than starting a service; there is
# no service yet.
CMD ["pytest", "tests/", "-q"]
