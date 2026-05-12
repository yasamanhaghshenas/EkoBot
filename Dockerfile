FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["sh", "-c", "python indexer.py && uvicorn main:app --host 0.0.0.0 --port 7860"]