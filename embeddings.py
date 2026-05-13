import hashlib
import math
import re


class SimpleHashEmbeddingFunction:
    def __init__(self, dim=256):
        self.dim = dim

    def name(self):
        return "simple_hash_embedding_v1"

    def _encode(self, texts):
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text):
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if (int(digest[8:10], 16) % 2 == 0) else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def __call__(self, input):
        if isinstance(input, str):
            return [self._embed_one(input)]
        return [self._embed_one(text) for text in input]

    def embed_documents(self, input):
        return self.__call__(input)

    def embed_query(self, input):
        return self.__call__(input)


class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = None
        self.fallback = SimpleHashEmbeddingFunction()
        
        # Keep the embedding model local. The project refreshes the vector
        # store ahead of each semester, so runtime should not depend on live
        # model downloads.
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[WARN] Could not load sentence-transformers model '{model_name}': {e}")
            print("[WARN] Falling back to SimpleHashEmbeddingFunction")
            self.model = None

    def name(self):
        return f"sentence_transformer:{self.model_name}"

    def _encode(self, texts):
        if self.model is None:
            # Use fallback if model failed to load
            return self.fallback._encode(texts)
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def __call__(self, input):
        if isinstance(input, str):
            return self._encode([input])
        return self._encode(list(input))

    def embed_documents(self, input):
        return self.__call__(input)

    def embed_query(self, input):
        return self.__call__(input)
