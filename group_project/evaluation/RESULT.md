# RAG evaluation results — Du lịch Hà Nội

## Run information

| Field | Value |
|---|---|
| Evaluation date | 2026-09-25T04:13:47.765762+00:00 |
| Framework and version | Ragas 0.4.3 |
| Evaluator model | gpt-4o |
| Generator model | gpt-4o |
| Embedding model | text-embedding-3-small |
| Corpus version/commit | 589fcc7 |
| Golden dataset size | 15 |
| `top_k` | 5 |
| Fallback threshold and calibration | 0.3; cần lưu riêng các query calibration khi đổi corpus |

## Configurations

- **Config A — dense-only:** embedding + cosine search, không BM25/RRF.
- **Config B — hybrid + RRF:** dense và BM25, fusion RRF một lần; PageIndex khi dense score dưới threshold.

Hai cấu hình dùng cùng golden dataset, generator, evaluator, prompt và `top_k`.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
| Faithfulness | 0.9778 | 0.9667 | -0.0111 |
| Answer relevance | 0.5866 | 0.6171 | +0.0305 |
| Context recall | 0.8000 | 0.8000 | +0.0000 |
| Context precision | 0.5208 | 0.5100 | -0.0108 |
| **Average** | **0.7213** | **0.7234** | **+0.0021** |

## A/B comparison

- Cấu hình tốt hơn: **hybrid + RRF**.
- Evidence: điểm trung bình 0.7234 trên 15 câu grounded.
- Trade-off về latency/cost: dense-only 36.820s/câu; hybrid + RRF 27.493s/câu.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | Sự kiện lịch sử nào từng diễn ra tại Nhà hát lớn Hà Nội? | dense-only | 1.000 | 0.000 | 0.000 | 0.000 | retrieval | Metric thấp nhất: answer_relevance |
| 2 | Bài viết về kiến trúc Pháp cổ nhắc đến những công trình tiêu biểu nào? | dense-only | 1.000 | 0.000 | 0.000 | 0.000 | retrieval | Metric thấp nhất: answer_relevance |
| 3 | Sự kiện lịch sử nào từng diễn ra tại Nhà hát lớn Hà Nội? | hybrid + RRF | 1.000 | 0.000 | 0.000 | 0.000 | retrieval | Metric thấp nhất: answer_relevance |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Tối ưu `context_precision` bằng kiểm tra ba câu tệ nhất | Đây là metric thấp nhất của hybrid (0.5100) | Tăng chất lượng tổng thể | Chạy lại đúng 15 câu và so delta |
| 2 | Hiệu chỉnh chunk size/overlap và threshold | Truy vấn địa danh và số liệu nằm ở nhiều loại tài liệu | Tăng recall, giảm fallback sai | Grid search trên golden dataset |
| 3 | Bổ sung nguồn giờ mở cửa/giá vé chính thức | Corpus hiện mạnh về lịch sử và quy hoạch hơn thông tin vận hành | Mở rộng phạm vi câu hỏi hữu ích | Thêm golden cases cho giờ/giá và chạy regression |

## Bonus experiments

Chưa chạy bonus. Chỉ triển khai HyDE, reranker nâng cao hoặc memory sau khi baseline ổn định.
