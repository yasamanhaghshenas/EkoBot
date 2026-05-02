FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model during build
RUN python -c "from chromadb.utils import embedding_functions; \
	ef = embedding_functions.SentenceTransformerEmbeddingFunction(\
	model_name='all-MiniLM-L6-v2'); print('Model downloaded!')"

COPY . .

RUN python startup.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
