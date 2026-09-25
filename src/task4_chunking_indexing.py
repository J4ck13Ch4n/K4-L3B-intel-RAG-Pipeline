"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
_DEFAULT_DIM = "768" if EMBEDDING_PROVIDER == "gemini" else "1024"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", _DEFAULT_DIM))

COLLECTION_NAME = "rag_documents"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text bằng provider được cấu hình trong môi trường."""
    if not texts:
        return []

    provider = EMBEDDING_PROVIDER.strip().lower()
    if provider == "sentence_transformers":
        model = _sentence_transformer(EMBEDDING_MODEL)
        vectors = model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]
    if provider == "openai":
        from openai import OpenAI

        response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]
    if provider == "gemini":
        from google import genai
        from google.genai import types

        response = genai.Client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        vectors: list[list[float]] = []
        for item in response.embeddings:
            values = [float(value) for value in item.values]
            norm = sum(value * value for value in values) ** 0.5
            vectors.append([value / norm for value in values] if norm else values)
        return vectors
    raise ValueError(f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER}")


@lru_cache(maxsize=2)
def _sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        front_matter, content = _parse_front_matter(raw)
        relative_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = front_matter.get(
            "doc_type", "legal" if "legal" in path.parts else "news"
        )
        documents.append(
            {
                "id": relative_id,
                "content": content.strip(),
                "metadata": {
                    "source": front_matter.get("source", path.name),
                    "title": front_matter.get("title", path.stem),
                    "doc_type": doc_type,
                    "url": front_matter.get("url") or None,
                },
            }
        )
    return documents


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    marker = text.find("\n---\n", 4)
    if marker < 0:
        return {}, text
    metadata: dict[str, str] = {}
    for line in text[4:marker].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, text[marker + 5 :]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )
        split_text = splitter.split_text
    except ImportError:
        split_text = _fallback_split_text
    chunks: list[dict] = []
    for document in documents:
        texts = split_text(document["content"])
        for index, text in enumerate(text for text in texts if text.strip()):
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text.strip(),
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def _fallback_split_text(text: str) -> list[str]:
    """Fallback không dependency; ưu tiên cắt ở ranh giới câu/khoảng trắng."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            window = text[start:end]
            candidates = [window.rfind(separator) for separator in ("\n\n", "\n", ". ", " ")]
            boundary = max(candidates)
            if boundary >= int(CHUNK_SIZE * 0.6):
                end = start + boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    batch_size = max(int(os.getenv("EMBEDDING_BATCH_SIZE", "32")), 1)
    texts = [chunk["content"] for chunk in chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embed_texts(texts[start : start + batch_size]))
        print(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks")
    if len(vectors) != len(chunks):
        raise ValueError("Embedding provider trả sai số lượng vector")
    return [
        {**chunk, "embedding": vector}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return
    collection = get_collection()
    metadatas = []
    for chunk in chunks:
        metadata = {
            key: ("" if value is None else value)
            for key, value in chunk["metadata"].items()
        }
        metadatas.append(metadata)
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=metadatas,
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
