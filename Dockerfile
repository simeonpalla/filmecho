FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ agents/
COPY orchestration/ orchestration/
COPY backend/ backend/

# Cloud Run sets $PORT; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}
