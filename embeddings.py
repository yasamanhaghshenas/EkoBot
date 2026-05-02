import chromadb.utils.embedding_functions as ef


def get_embedding_function():
    return ef.DefaultEmbeddingFunction()
