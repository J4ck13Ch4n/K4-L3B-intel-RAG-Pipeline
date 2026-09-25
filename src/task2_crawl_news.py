"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://sodulich.hanoi.gov.vn/ve-mien-huyen-tich-co-loa.html",
    "https://sodulich.hanoi.gov.vn/pho-ha-noi.html",
    "https://sodulich.hanoi.gov.vn/ve-dep-nhung-cong-trinh-kien-truc-phap-co-tieu-bieu-tai-ha-noi.html",
    "https://sodulich.hanoi.gov.vn/nhung-xu-huong-du-lich-moi-noi-tai-viet-nam.html",
    "https://sodulich.hanoi.gov.vn/co-mot-sac-tim-nhu-ua-vao-long-pho-ha-noi-trong-nhung-ngay-thang-5.html",
]


class ArticleParser(HTMLParser):
    """Trích title và nội dung đọc được mà không phụ thuộc CSS của website."""

    BLOCK_TAGS = {"h1", "h2", "h3", "p", "li"}
    IGNORED_TAGS = {"script", "style", "svg", "nav", "footer", "form"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._title_parts: list[str] = []
        self._current_tag = ""
        self._current_parts: list[str] = []
        self._ignored_depth = 0
        self.blocks: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
        if not self._ignored_depth and tag in self.BLOCK_TAGS:
            self._current_tag = tag
            self._current_parts = []
        if tag == "title":
            self._title_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self._title_parts:
            self.title = " ".join(self._title_parts).strip()
        if not self._ignored_depth and tag == self._current_tag:
            text = re.sub(r"\s+", " ", " ".join(self._current_parts)).strip()
            if len(text) >= 20:
                self.blocks.append((tag, text))
            self._current_tag = ""
            self._current_parts = []
        if tag in self.IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._current_tag:
            self._current_parts.append(text)
        self._title_parts.append(text)

    def as_markdown(self) -> str:
        seen: set[str] = set()
        lines: list[str] = []
        for tag, text in self.blocks:
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            if tag.startswith("h"):
                level = min(int(tag[1]), 3)
                lines.append(f"{'#' * level} {text}")
            elif tag == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)
        return "\n\n".join(lines)


def _crawl_article_sync(url: str) -> dict:
    response = requests.get(
        url,
        headers={"User-Agent": "HanoiTourismRAG/1.0 (educational project)"},
        timeout=45,
        verify=os.getenv("ALLOW_INSECURE_SSL", "0") != "1",
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"

    parser = ArticleParser()
    parser.feed(response.text)
    markdown = parser.as_markdown()
    if len(markdown) < 200:
        raise ValueError(f"Không trích được đủ nội dung từ {url}")
    return {
        "url": url,
        "title": parser.title or url.rsplit("/", 1)[-1].removesuffix(".html"),
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": markdown,
    }


async def crawl_article(url: str) -> dict:
    """Crawl một bài mà không chặn event loop."""
    return await asyncio.to_thread(_crawl_article_sync, url)


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
