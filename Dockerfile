# DICOMweb AI Gateway — self-hosted, CPU-only by default.
FROM python:3.12-slim

WORKDIR /app

# Non-root user for the containerized gateway.
RUN useradd --create-home --uid 1000 gateway \
    && mkdir -p /app/data && chown -R gateway:gateway /app
USER gateway

COPY --chown=gateway:gateway pyproject.toml README.md LICENSE ./
COPY --chown=gateway:gateway src ./src

RUN pip install --no-cache-dir .

# Bind to the container interface so the port can be published.
ENV STORE_DIR=/app/data/store

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "dicomweb_ai_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
