FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/data && chown -R appuser:appuser /app
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
USER appuser
ENV PORT=8000
ENV INDEX_PATH=/app/data/index.bin
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn findex.web.app:app --host 0.0.0.0 --port ${PORT}"]
