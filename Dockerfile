FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# server/app.py is the FastAPI app; models.py is at root (on sys.path via app.py)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
