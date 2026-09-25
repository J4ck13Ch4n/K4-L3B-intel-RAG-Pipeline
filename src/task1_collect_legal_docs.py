"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

import os
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Nguồn công khai của Cổng thông tin điện tử Thành phố Hà Nội. Tên file ổn
# định để ID chunk không thay đổi giữa các lần chạy pipeline.
LEGAL_SOURCES = {
    "quyet-dinh-76-2025-dinh-muc-du-lich-ha-noi.pdf": (
        "https://datafiles.hanoi.gov.vn/gov-hni/6249/VanBanCongBao/2026/1/21/"
        "QDPQ-76-2025_signed.pdf"
    ),
    "de-an-cong-nghiep-van-hoa-ha-noi-2025.pdf": (
        "https://datafiles.hanoi.gov.vn/gov-hni/6244/VanBan/2025/10/27/"
        "PLQD-5287-2025.pdf"
    ),
    "quy-hoach-thu-do-ha-noi-2026.pdf": (
        "https://datafiles.hanoi.gov.vn/gov-hni/6244/VanBan/2026/5/14/"
        "QD-2512-2026.pdf"
    ),
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    setup_directory()
    headers = {"User-Agent": "HanoiTourismRAG/1.0 (educational project)"}

    for filename, url in LEGAL_SOURCES.items():
        destination = DATA_DIR / filename
        if destination.exists() and destination.stat().st_size > 1024:
            print(f"Exists: {destination}")
            continue

        verify_ssl = os.getenv("ALLOW_INSECURE_SSL", "0") != "1"
        response = requests.get(url, headers=headers, timeout=60, verify=verify_ssl)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"Nguồn không trả về PDF hợp lệ: {url}")
        destination.write_bytes(response.content)
        print(f"Saved: {destination} ({len(response.content):,} bytes)")


if __name__ == "__main__":
    setup_directory()
    download_documents()
