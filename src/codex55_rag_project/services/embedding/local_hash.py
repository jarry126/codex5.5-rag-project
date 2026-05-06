"""\u672c\u5730\u54c8\u5e0c\u5411\u91cf\u5316\u5668\uff08\u7528\u4e8e\u6d4b\u8bd5\uff09\u3002

\u4e2d\u6587\uff1a\u4f7f\u7528 MD5 \u54c8\u5e0c\u751f\u6210\u786e\u5b9a\u6027\u5411\u91cf\uff0c\u4e0d\u8c03\u7528\u5916\u90e8\u6a21\u578b\uff0c\u4fdd\u8bc1\u6d4b\u8bd5\u53ef\u8fd0\u884c\u4e14\u7ed3\u679c\u53ef\u9884\u6d4b\u3002
English: Uses MD5 hash to generate deterministic vectors, no external model calls, ensures tests run with predictable results.
"""

from __future__ import annotations

import hashlib
import math
import re

from codex55_rag_project.core.ports import Vector


# \u5339\u914d\u4e2d\u82f1\u6587\u8bcd\u8bed\u7684\u6b63\u5219\u8868\u8fbe\u5f0f
TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class HashingEmbedder:
    """Deterministic local embedder for tests and demos.

    \u4e2d\u6587\uff1a\u4f7f\u7528 MD5 \u54c8\u5e0c\u5c06\u8bcd\u8bed\u6620\u5c04\u5230\u5411\u91cf\u7ef4\u5ea6\uff0c\u751f\u6210\u786e\u5b9a\u6027\u5411\u91cf\u7528\u4e8e\u6d4b\u8bd5\u3002
    English: Uses MD5 hash to map words to vector dimensions, generating deterministic vectors for tests.
    Replace this class with a model-backed embedder in production.
    """

    def __init__(self, dimensions: int = 256) -> None:
        # dimensions \u63a7\u5236\u5411\u91cf\u7ef4\u5ea6\uff0c\u8d8a\u5927\u5411\u91cf\u8d8a\u7a00\u758f\u4f46\u533a\u5206\u5ea6\u8d8a\u9ad8\u3002
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        """\u6279\u91cf\u6587\u672c\u5411\u91cf\u5316\u3002

        \u4e2d\u6587\uff1a\u5bf9\u6bcf\u4e2a\u6587\u672c\u8c03\u7528 _embed \u751f\u6210\u5411\u91cf\u3002
        English: Calls _embed for each text to generate vectors.
        """
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> Vector:
        """\u751f\u6210\u5355\u4e2a\u6587\u672c\u7684\u5411\u91cf\u3002

        \u4e2d\u6587\uff1a\u63d0\u53d6\u8bcd\u8bed\uff0c\u7528 MD5 \u54c8\u5e0c\u51b3\u5b9a\u5411\u91cf\u7684\u4f4d\u7f6e\u548c\u7b26\u53f7\uff0c\u6700\u540e\u5f52\u4e00\u5316\u3002
        English: Extracts words, uses MD5 hash to determine position and sign, then normalizes.
        """
        # \u521d\u59cb\u5316\u96f6\u5411\u91cf
        vector = [0.0] * self.dimensions
        # \u5bf9\u6bcf\u4e2a\u8bcd\u8bed\uff1a\u54c8\u5e0c\u51b3\u5b9a\u7ef4\u5ea6\u4f4d\u7f6e\uff0c\u7b2c 5 \u5b57\u8282\u51b3\u5b9a\u6b63\u8d1f\u53f7
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        # L2 \u5f52\u4e00\u5316\uff0c\u4f7f\u5411\u91cf\u53ef\u7528\u4e8e cosine similarity
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

