"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


import math
import re
from collections import Counter


CORPUS: list[dict] = []


def tokenize(text: str) -> list[str]:
    """Tokenizer Unicode nhỏ gọn, giữ được từ tiếng Việt và mã/số."""
    return re.findall(r"[\wÀ-ỹ]+", text.casefold(), flags=re.UNICODE)


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    return BM25Index([tokenize(item["content"]) for item in corpus])


class BM25Index:
    """BM25 nhỏ gọn với IDF dương, đủ để chạy offline và tái lập kết quả."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.frequencies = [Counter(document) for document in documents]
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        count = len(documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for frequencies, length in zip(self.frequencies, self.lengths):
            score = 0.0
            normalization = self.k1 * (
                1.0 - self.b + self.b * length / max(self.average_length, 1.0)
            )
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + normalization)
                )
            scores.append(score)
        return scores


def _get_corpus() -> list[dict]:
    if CORPUS:
        return CORPUS
    from .task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    corpus = _get_corpus()
    query_tokens = tokenize(query)
    if top_k <= 0 or not corpus or not query_tokens:
        return []
    scores = build_bm25_index(corpus).get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(corpus)), key=lambda index: float(scores[index]), reverse=True
    )
    results: list[dict] = []
    seen_ids: set[str] = set()
    for index in ranked_indices:
        score = float(scores[index])
        item = corpus[index]
        if score <= 0 or item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
