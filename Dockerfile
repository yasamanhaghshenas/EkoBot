FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from chromadb.utils import embedding_functions; \
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(\
    model_name='all-MiniLM-L6-v2'); print('Model ready!')"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
