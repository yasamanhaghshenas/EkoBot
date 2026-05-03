FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import chromadb; client = chromadb.Client(); collection = client.create_collection('test'); collection.add(documents=['test'], ids=['1']); collection.query(query_texts=['test'], n_results=1); print('Model pre-downloaded!'); client.delete_collection('test')"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
