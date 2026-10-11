# Build from the repository root (the app imports ingestion/symbol_config.py):
#   docker build -f flask_app/Dockerfile -t stock-dashboard .
FROM python:3.11-slim

WORKDIR /app

COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app/ .
COPY ingestion/symbol_config.py .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 60 app:app"]
