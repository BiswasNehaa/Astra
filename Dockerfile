FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY . .

# Run as a non-root user rather than the container default (root).
# data/chroma/ is written to at runtime, so chown happens after COPY.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]