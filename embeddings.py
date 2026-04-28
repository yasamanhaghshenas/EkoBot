import hashlib
import math
import re


class SimpleHashEmbeddingFunction:
    def __init__(self, dim=256):
        self.dim = dim

    def name(self):
        return "simple_hash_embedding_v1"

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
        self.model_name = model_name
        self._fallback = SimpleHashEmbeddingFunction()
        self._using_fallback = False

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            print("WARNING: sentence-transformers is not installed; using local hash embeddings.")
            self.model = None
            self._using_fallback = True
            return

        try:
            # Keep the embedding model local. The project refreshes the vector
            # store ahead of each semester, so runtime should not depend on live
            # model downloads.
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception as exc:
            print(
                f"WARNING: Could not load sentence-transformers model '{model_name}' locally; "
                "falling back to local hash embeddings."
            )
            self.model = None
            self._using_fallback = True
            self._fallback_error = exc

    def name(self):
        if self._using_fallback:
            return self._fallback.name()
        return f"sentence_transformer:{self.model_name}"

    def _encode(self, texts):
        if self._using_fallback or self.model is None:
            return self._fallback._embed_one(texts[0]) if isinstance(texts, list) and len(texts) == 1 else self._fallback(texts)
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def __call__(self, input):
        if self._using_fallback or self.model is None:
            return self._fallback(input)
        if isinstance(input, str):
            return self._encode([input])
        return self._encode(list(input))

    def embed_documents(self, input):
        return self.__call__(input)

    def embed_query(self, input):
        return self.__call__(input)
