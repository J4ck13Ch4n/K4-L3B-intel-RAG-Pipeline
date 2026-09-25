"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
import re
from pathlib import Path

from .task1_collect_legal_docs import LEGAL_SOURCES


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert PDF/DOCX và gắn metadata nguồn vào front matter."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        result = converter.convert(str(path))
        content = result.text_content.strip()
        if not content:
            raise ValueError(f"Kết quả chuyển đổi rỗng: {path}")
        title = _humanize_title(path.stem)
        source_url = LEGAL_SOURCES.get(path.name, "")
        header = _front_matter(
            source=path.name,
            title=title,
            doc_type="legal",
            url=source_url,
        )
        destination = output_dir / f"{path.stem}.md"
        destination.write_text(header + f"# {title}\n\n" + content, encoding="utf-8")
        print(f"Saved: {destination}")


def convert_news_articles() -> None:
    """Chuẩn hóa JSON bài viết thành Markdown có metadata thống nhất."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"url", "title", "date_crawled", "content_markdown"}

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - data.keys()
        if missing:
            raise ValueError(f"{path.name} thiếu metadata: {sorted(missing)}")
        content = str(data["content_markdown"]).strip()
        if not content:
            raise ValueError(f"Nội dung rỗng: {path}")
        header = _front_matter(
            source=path.name,
            title=str(data["title"]),
            doc_type="news",
            url=str(data["url"]),
            crawled=str(data["date_crawled"]),
        )
        destination = output_dir / f"{path.stem}.md"
        destination.write_text(
            header + f"# {data['title']}\n\n" + content,
            encoding="utf-8",
        )
        print(f"Saved: {destination}")


def _humanize_title(stem: str) -> str:
    return re.sub(r"[-_]+", " ", stem).strip().title()


def _front_matter(
    *, source: str, title: str, doc_type: str, url: str, crawled: str = ""
) -> str:
    values = {
        "source": source,
        "title": title.replace('"', "'"),
        "doc_type": doc_type,
        "url": url,
    }
    if crawled:
        values["date_crawled"] = crawled
    body = "\n".join(f'{key}: "{value}"' for key, value in values.items())
    return f"---\n{body}\n---\n\n"


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
