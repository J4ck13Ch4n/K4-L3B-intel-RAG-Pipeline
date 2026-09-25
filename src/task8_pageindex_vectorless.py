"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.

API của SDK pageindex==0.2.8 (chỉ nhận PDF):
    PageIndexClient(api_key)
    submit_document(path)       -> {"doc_id": ...}
    is_retrieval_ready(doc_id)  -> bool
    submit_query(doc_id, query) -> {"retrieval_id": ...}
    get_retrieval(retrieval_id) -> {"status": ..., "retrieved_nodes": [...]}

Response thật (09/2026): mỗi node có "id", "title" và "relevant_contents" dạng
list lồng list các {"section_title", "physical_index", "relevant_content"}.
Endpoint retrieval đã được PageIndex đánh dấu deprecated (khuyên dùng chat API)
nhưng vẫn hoạt động.

Chạy một lần để upload (PageIndex cần vài phút xử lý mỗi PDF):
    python -m src.task8_pageindex_vectorless
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from .task1_collect_legal_docs import LEGAL_SOURCES


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CACHE_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"

QUERY_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 1.5


def _client():
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> None:
    """Upload PDF legal và lưu mapping filename -> doc_id để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY")
    client = _client()
    cache = _load_cache()
    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        if path.name not in cache:
            cache[path.name] = client.submit_document(str(path))["doc_id"]
            CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            print(f"Uploaded: {path.name} -> {cache[path.name]}")
    for filename, doc_id in cache.items():
        ready = client.is_retrieval_ready(doc_id)
        print(f"{filename}: {'ready' if ready else 'processing'}")


def _query_document(client, doc_id: str, query: str, deadline: float) -> list[dict]:
    retrieval_id = client.submit_query(doc_id, query)["retrieval_id"]
    while time.monotonic() < deadline:
        response = client.get_retrieval(retrieval_id)
        status = response.get("status")
        if status == "completed":
            return response.get("retrieved_nodes") or []
        if status == "failed":
            return []
        time.sleep(POLL_INTERVAL_SECONDS)
    return []


def _node_text(node: dict) -> str:
    """Gộp relevant_content; API trả list lồng list: [[{"relevant_content": ...}], ...]."""

    def walk(value):
        if isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif isinstance(value, dict):
            yield str(value.get("relevant_content") or "")
        elif isinstance(value, str):
            yield value

    texts = (text.strip() for text in walk(node.get("relevant_contents") or []))
    return "\n\n".join(text for text in texts if text)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult; score giảm dần theo thứ tự PageIndex trả về."""
    if top_k <= 0 or not query.strip() or not PAGEINDEX_API_KEY:
        return []
    cache = _load_cache()
    if not cache:
        return []

    client = _client()
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
    with ThreadPoolExecutor(max_workers=len(cache)) as pool:
        futures = {
            filename: pool.submit(_query_document, client, doc_id, query, deadline)
            for filename, doc_id in cache.items()
        }
        nodes_by_file: dict[str, list[dict]] = {}
        for filename, future in futures.items():
            try:
                nodes_by_file[filename] = future.result(timeout=QUERY_TIMEOUT_SECONDS + 5)
            except Exception as error:
                print(f"PageIndex query failed for {filename}: {error}")

    # Xen kẽ node của các tài liệu để một văn bản dài không chiếm hết top_k.
    results: list[dict] = []
    seen: set[str] = set()
    for rank in range(max((len(nodes) for nodes in nodes_by_file.values()), default=0)):
        for filename, nodes in nodes_by_file.items():
            if rank >= len(nodes):
                continue
            node = nodes[rank]
            content = _node_text(node)
            node_id = node.get("id") or node.get("node_id") or rank
            item_id = f"pageindex::{cache[filename]}::{node_id}"
            if not content or item_id in seen:
                continue
            seen.add(item_id)
            title = Path(filename).stem.replace("-", " ").title()
            results.append(
                {
                    "id": item_id,
                    "content": content,
                    "score": 0.0,
                    "metadata": {
                        "source": filename,
                        "title": f"{title} — {node['title']}" if node.get("title") else title,
                        "doc_type": "legal",
                        "url": LEGAL_SOURCES.get(filename),
                        "chunk_index": len(results),
                    },
                    "retrieval_method": "pageindex",
                }
            )
    results = results[:top_k]
    for rank, item in enumerate(results, 1):
        item["score"] = 1.0 / rank
    return results


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    upload_documents()
