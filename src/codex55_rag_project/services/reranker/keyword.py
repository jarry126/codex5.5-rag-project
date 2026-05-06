"""\u5173\u952e\u8bcd\u91cd\u6392\u5e8f\u5668\u3002

\u4e2d\u6587\uff1a\u57fa\u4e8e\u5173\u952e\u8bcd\u91cd\u53e0\u5ea6\u91cd\u6392\u5e8f\uff0c\u7528\u4e8e\u65e0\u5916\u90e8 API \u65f6\u7684\u672c\u5730\u964d\u7ea7\u65b9\u6848\u3002
English: Reranks based on keyword overlap, used as local fallback when external API is unavailable.
"""

from __future__ import annotations

import re

from codex55_rag_project.core.models import RetrievedChunk


# \u5339\u914d\u4e2d\u6587\u548c\u82f1\u6587\u5355\u8bcd\u7684\u6b63\u5219\u8868\u8fbe\u5f0f
TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class NoopReranker:
    """\u7a7a\u91cd\u6392\u5e8f\u5668\u3002

    \u4e2d\u6587\uff1a\u4e0d\u6539\u53d8\u987a\u5e8f\uff0c\u76f4\u63a5\u8fd4\u56de\u524d top_k \u4e2a\u5019\u9009\u3002
    English: No reranking, directly returns first top_k candidates.
    """

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """\u4e0d\u505a\u91cd\u6392\u5e8f\uff0c\u76f4\u63a5\u622a\u53d6\u524d top_k\u3002"""
        return candidates[:top_k]


class KeywordOverlapReranker:
    """\u5173\u952e\u8bcd\u91cd\u53e0\u91cd\u6392\u5e8f\u5668\u3002

    \u4e2d\u6587\uff1a\u8ba1\u7b97\u67e5\u8be2\u548c\u5019\u9009\u5207\u7247\u7684\u8bcd\u8bed\u91cd\u53e0\u5ea6\uff0c\u5c06\u91cd\u53e0\u5ea6\u52a0\u5230\u539f\u59cb score \u4e0a\u91cd\u6392\u5e8f\u3002
    English: Computes word overlap between query and candidate, adds overlap to original score for reranking.
    """

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """\u57fa\u4e8e\u5173\u952e\u8bcd\u91cd\u53e0\u5ea6\u91cd\u6392\u5e8f\u3002

        \u4e2d\u6587\uff1a\u63d0\u53d6\u67e5\u8be2\u4e2d\u7684\u8bcd\u8bed\uff0c\u8ba1\u7b97\u4e0e\u6bcf\u4e2a\u5019\u9009\u5207\u7247\u7684\u91cd\u53e0\u6570\uff0c\u53e0\u52a0\u5230 score \u4e0a\u3002
        English: Extracts query words, computes overlap with each candidate, adds to score.
        """
        # \u63d0\u53d6\u67e5\u8be2\u4e2d\u7684\u8bcd\u8bed\uff08\u4e2d\u82f1\u6587\uff09
        query_terms = set(TOKEN_PATTERN.findall(query.lower()))

        def score(candidate: RetrievedChunk) -> float:
            # \u63d0\u53d6\u5019\u9009\u5207\u7247\u4e2d\u7684\u8bcd\u8bed\uff0c\u8ba1\u7b97\u91cd\u53e0\u6570
            chunk_terms = set(TOKEN_PATTERN.findall(candidate.chunk.text.lower()))
            overlap = len(query_terms & chunk_terms)
            # \u6bcf\u4e2a\u91cd\u53e0\u8bcd\u589e\u52a0 0.05 \u5206\uff0c\u53e0\u52a0\u539f\u59cb\u76f8\u4f3c\u5ea6\u5206\u6570
            return candidate.score + overlap * 0.05

        # \u6309\u65b0 score \u964d\u5e8f\u6392\u5e8f\uff0c\u8fd4\u56de top_k
        return sorted(candidates, key=score, reverse=True)[:top_k]

