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