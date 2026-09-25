# Hà Nội Travel Assistant — RAG Pipeline

## Mục tiêu

Chatbot RAG tiếng Việt trả lời câu hỏi về điểm đến, di sản, ẩm thực và định hướng phát triển du lịch Hà Nội. Sản phẩm dùng hybrid retrieval, citation và giao diện Streamlit.

Corpus hiện gồm ba văn bản chính thức của Thành phố Hà Nội và năm bài cẩm nang từ Sở Du lịch Hà Nội. Mỗi câu trả lời chỉ được xác nhận khi có citation `[S1]`, `[S2]` đối chiếu được với nguồn hiển thị trên giao diện.

## Phạm vi câu hỏi

- Điểm đến và di sản: Cổ Loa, khu vực trung tâm, các công trình kiến trúc Pháp.
- Ẩm thực: phở Hà Nội và các địa chỉ được nguồn công khai giới thiệu.
- Trải nghiệm theo mùa và du lịch xanh.
- Quy hoạch, định hướng và dịch vụ quảng bá du lịch Hà Nội.
- Câu hỏi ngoài corpus được trả lời bằng safe refusal, không suy đoán.

## Sản phẩm phải nộp

- Repository nhóm chạy được.
- Tối thiểu 3 tài liệu chính sách và 5 bài viết/page do nhóm tự thu thập.
- Pipeline: convert → chunk → index → dense + BM25 → RRF → fallback → generation có citation.
- Chatbot Streamlit hiển thị câu trả lời và nguồn đã dùng.
- Golden dataset tối thiểu 15 câu; đánh giá 4 metric và so sánh A/B.
- `group_project/evaluation/RESULT.md`.
- Mỗi thành viên nộp báo cáo cá nhân theo template trong `reports/INDIVIDUAL_REPORT.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Điền API key và model cần dùng trong `.env`; không commit file này. Cấu hình đã được kiểm thử với Gemini cho generation và embedding. Chỉ đặt `ALLOW_INSECURE_SSL=1` khi máy gặp lỗi chuỗi chứng chỉ với website nguồn.

```bash
# 1. Thu thập và chuẩn hoá
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index và kiểm tra contract
python -m src.task4_chunking_indexing
pytest -q

# 3. Chạy sản phẩm
streamlit run app.py
```

## Lộ trình 3 giờ

| Mốc                  | Thời gian | Kết quả cần có                           |
| -------------------- | --------: | ---------------------------------------- |
| 0. Setup             |   10 phút | Môi trường và `.env` sẵn sàng            |
| 1. Data              |   25 phút | ≥3 legal, ≥5 news, Markdown đã chuẩn hoá |
| 2. Index & search    |   30 phút | ChromaDB, dense search và BM25 chạy được |
| 3. Fusion & fallback |   25 phút | RRF và fallback tuân thủ contract        |
| 4. Generation & UI   |   30 phút | Chatbot trả lời có citation              |
| 5. Evaluation        |   30 phút | 15+ Q&A, 4 metric, A/B comparison        |
| 6. Demo & handoff    |   30 phút | Test, report, demo và push repository    |

## Lưu ý quy tắc để có code quality tốt:

- Dense và BM25 nên cùng trả về `SearchResult` theo một schema.
- RRF chỉ nên dùng để gộp thứ hạng và chỉ chạy một lần.
- Fallback dùng cosine score gốc của dense retrieval.
- Threshold phải được hiệu chỉnh trên query in domain và out of domain, không có một con số đúng cho mọi corpus.

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md): schema, interface và invariant mà code/test nên tuân theo.
- [Step-by-step guide](docs/STEP_BY_STEP.md): thứ tự triển khai và tiêu chí hoàn thành từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md): Rubric thang điểm.
- [Individual report](reports/INDIVIDUAL_REPORT.md): template báo cáo cá nhân.
- [Suggested topics](docs/SUGGESTED_TOPICS.md): danh sách chủ đề tham khảo, không bắt buộc.

## Kiểm tra

```bash
# Contract tests
pytest tests/test_contracts.py -q

# Acceptance tests
pytest tests/test_acceptance.py -q

# Toàn bộ
pytest -q
```
