"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"
WORKSPACE_DIR = Path(__file__).parent.parent / ".pageindex_workspace"


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY")
    from pageindex import PageIndexClient

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY, workspace=str(WORKSPACE_DIR))
    cache = _load_cache()
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        source = path.relative_to(STANDARDIZED_DIR).as_posix()
        if source in cache:
            continue
        document_id = client.index(str(path), mode="auto")
        cache[source] = {"doc_id": document_id, "path": str(path)}
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Indexed PageIndex: {source} -> {document_id}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if top_k <= 0 or not query.strip() or not PAGEINDEX_API_KEY:
        return []
    cache = _load_cache()
    if not cache:
        upload_documents()
        cache = _load_cache()

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY, workspace=str(WORKSPACE_DIR))
    query_terms = set(_tokenize(query))
    candidates: list[dict] = []
    for source, entry in cache.items():
        structure = client.get_document_structure(entry["doc_id"])
        if isinstance(structure, str):
            structure = json.loads(structure)
        for node_index, node in enumerate(_flatten_nodes(structure)):
            content = str(node.get("text") or node.get("summary") or "").strip()
            if not content:
                continue
            title = str(node.get("title") or Path(source).stem)
            terms = set(_tokenize(f"{title} {content}"))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            if overlap <= 0:
                continue
            candidates.append(
                {
                    "id": f"pageindex::{entry['doc_id']}::{node.get('node_id', node_index)}",
                    "content": content,
                    "score": float(overlap),
                    "metadata": {
                        "source": source,
                        "title": title,
                        "doc_type": "legal" if source.startswith("legal/") else "news",
                        "url": "",
                        "chunk_index": node_index,
                    },
                    "retrieval_method": "pageindex",
                }
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.casefold(), flags=re.UNICODE)


def _flatten_nodes(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_nodes(item)
    elif isinstance(value, dict):
        if any(key in value for key in ("title", "text", "summary")):
            yield value
        for key in ("nodes", "children"):
            if key in value:
                yield from _flatten_nodes(value[key])


if __name__ == "__main__":
    upload_documents()
