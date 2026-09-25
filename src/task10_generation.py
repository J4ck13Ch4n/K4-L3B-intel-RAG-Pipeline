"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định thực tế phải có citation dạng [S1], [S2].
Không sử dụng kiến thức ngoài context. Nếu thiếu evidence, hãy từ chối xác minh.
Trả lời bằng tiếng Việt, ngắn gọn và hữu ích cho khách du lịch."""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ các nguồn du lịch Hà Nội hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict], source_ids: list[str] | None = None) -> str:
    """Tạo context có title và source label.

    ``source_ids`` là thứ tự của ``sources`` trả cho người dùng (theo score);
    nhãn [S<n>] lấy theo vị trí trong đó để citation map đúng về sources dù
    chunks đã được reorder.
    """
    order = source_ids or [chunk["id"] for chunk in chunks]
    parts: list[str] = []
    for chunk in chunks:
        index = order.index(chunk["id"]) + 1
        metadata = chunk["metadata"]
        parts.append(
            f"[S{index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']} | URL: {metadata.get('url') or 'N/A'}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    if not LLM_MODEL:
        raise ValueError("Cần cấu hình LLM_MODEL trong .env")
    provider = LLM_PROVIDER.strip().lower()
    if provider == "openai":
        from openai import OpenAI

        response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).responses.create(
            model=LLM_MODEL,
            instructions=system_prompt,
            input=user_message,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.output_text.strip()
    if provider == "gemini":
        from google import genai
        from google.genai import types

        # Giữ biến client sống suốt request; client tạm thời có thể bị GC đóng.
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return (response.text or "").strip()
    if provider == "anthropic":
        from anthropic import Anthropic

        response = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model=LLM_MODEL,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=900,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return "".join(block.text for block in response.content if hasattr(block, "text")).strip()
    raise ValueError(f"LLM_PROVIDER không hỗ trợ: {LLM_PROVIDER}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        chunks = []
    if not chunks:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}

    answer = answer_from_chunks(query, chunks)
    valid_citations = {
        int(number)
        for number in re.findall(r"\[S(\d+)\]", answer)
        if 1 <= int(number) <= len(chunks)
    }
    if not answer or SAFE_REFUSAL in answer or not valid_citations:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    return {
        "answer": answer,
        # Giữ thứ tự theo score để [S<n>] ứng với sources[n-1].
        "sources": chunks,
        "retrieval_source": (
            "pageindex" if chunks[0]["retrieval_method"] == "pageindex" else "hybrid"
        ),
    }


def answer_from_chunks(query: str, chunks: list[dict]) -> str:
    """Reorder, format context và gọi LLM; lỗi provider trả chuỗi rỗng."""
    context = format_context(reorder_for_llm(chunks), [chunk["id"] for chunk in chunks])
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        return call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:
        print(f"LLM provider error: {error}")
        return ""


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Nhà hát lớn Hà Nội được xây dựng vào năm nào?"
    result = generate_with_citation(question)
    print(f"Q: {question}\n\n{result['answer']}\n\nretrieval_source: {result['retrieval_source']}")
    for number, source in enumerate(result["sources"], 1):
        print(
            f"[S{number}] {source['metadata']['title']} · {source['retrieval_method']} "
            f"· score={source['score']:.4f} · {source['id']}"
        )
