# RAG evaluation results — Du lịch Hà Nội

## Run information

| Field | Value |
|---|---|
| Evaluation date | 2026-09-25T04:13:47+00:00 |
| Framework and version | Ragas 0.4.3 (collections API: `Faithfulness`, `AnswerRelevancy`, `ContextRecall`, `ContextPrecision`) |
| Evaluator model | gpt-4o (OpenAI, `AsyncOpenAI` client) |
| Generator model | gpt-4o, temperature 0.3, top_p 0.9 |
| Embedding model | text-embedding-3-small (1536 chiều), ChromaDB cosine |
| Corpus version/commit | 589fcc7 — 3 văn bản legal (PDF, datafiles.hanoi.gov.vn) + 5 bài Sở Du lịch Hà Nội → 1376 chunks (500 ký tự, overlap 50) |
| Golden dataset size | 15 câu, mỗi câu có `expected_answer` và `expected_context` trích từ corpus |
| `top_k` | 5 (dense và BM25 lấy 10 ứng viên trước khi fuse) |
| Fallback threshold and calibration | Lúc chạy: 0.3. Sau đó hiệu chỉnh lại thành **0.45**: best dense cosine của 15 câu golden nằm trong 0.471–0.737, của 5 câu ngoài domain (nấu ăn, bitcoin, bóng đá, cài Python, thời tiết Paris) nằm trong 0.254–0.421. Với cả hai ngưỡng, không câu golden nào kích hoạt PageIndex fallback, nên số liệu A/B dưới đây không bị ảnh hưởng. |

Dữ liệu chi tiết từng câu (answer, source IDs, điểm, latency): [`raw_results.json`](../group_project/evaluation/raw_results.json).

## Configurations

- **Config A — dense-only:** `semantic_search` với text-embedding-3-small, lấy top 5 theo cosine, không BM25/RRF.
- **Config B — hybrid + RRF:** dense top 10 và BM25 top 10, fuse đúng một lần bằng RRF (k = 60) rồi lấy top 5; PageIndex fallback khi dense cosine tốt nhất dưới threshold.

Hai config dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
| Faithfulness | 0.9778 | 0.9667 | -0.0111 |
| Answer relevance | 0.5866 | 0.6171 | +0.0305 |
| Context recall | 0.8000 | 0.8000 | +0.0000 |
| Context precision | 0.5208 | 0.5100 | -0.0108 |
| **Average** | **0.7213** | **0.7234** | **+0.0021** |

## A/B comparison

- **Cấu hình tốt hơn:** hybrid + RRF, nhưng chỉ nhỉnh hơn **+0.0021**. Với 15 câu, mức chênh này nằm trong nhiễu của LLM judge, nên chưa đủ để kết luận hybrid vượt trội.
- **Evidence theo từng câu** (điểm trung bình 4 metric):
  - Hybrid thắng ở 5 câu (#1, #8, #11, #13, #15). Rõ nhất: #13 "bốn trung tâm du lịch" (+0.19, precision 0.20 → 1.00) nhờ BM25 khớp cụm "04 trung tâm động lực"; #8 (+0.08) nhờ kéo thêm `article_03::chunk-1`.
  - Dense thắng ở 5 câu (#2, #3, #5, #9, #14). Rõ nhất: #3 (−0.09, faithfulness 0.67 → 0.50) và #9 (−0.08, precision 0.83 → 0.45). BM25 đẩy các chunk chứa từ phổ biến ("Cổ Loa", "du lịch") lên trên, làm context precision giảm.
  - Hoà ở 5 câu (#4, #6, #7, #10, #12), trong đó #7 bằng 0.25 ở cả hai config (xem Worst performers). Nhìn theo từng câu, hai config ngang nhau (5–5–5).
  - Context recall bằng nhau (0.80): cả hai config cùng trượt 3 câu #7, #8, #13. Nguyên nhân là chunking và dữ liệu, không phải chiến lược xếp hạng.
- **Trade-off về latency/cost:** latency đo được (dense-only 36.8 s/câu, hybrid 27.5 s/câu) **gồm cả thời gian 4 metric Ragas gọi gpt-4o**, nên không phản ánh chi phí retrieval. Phần retrieval riêng: BM25 in-memory và RRF chỉ thêm vài chục ms trên 1376 chunks, không tốn thêm API call. Chi phí thật của hybrid là bộ nhớ để giữ BM25 index, và phải index lại khi corpus đổi.
- **Lưu ý:** lượt chạy này gửi context cho Ragas theo thứ tự **đã reorder** (chống lost-in-the-middle). Context precision phụ thuộc thứ hạng nên số trên hơi lệch. `src/evaluate.py` đã được sửa để chấm trên thứ tự retrieval gốc, đo latency chỉ gồm retrieval + generation, và retry khi judge timeout. Lệch này áp dụng như nhau cho cả hai config nên không đổi kết luận A/B. Chạy lại `python -m src.evaluate` sẽ ghi đè file báo cáo này.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | Sự kiện lịch sử nào từng diễn ra tại Nhà hát lớn Hà Nội? (#7) | cả hai (dense-only) | 1.000 | 0.000 | 0.000 | 0.000 | retrieval (chunking) | Đáp án nằm trọn trong `article_03::chunk-2`, một chunk 150 ký tự bắt đầu bằng ". Không chỉ có giá trị… Đây là nơi diễn ra cuộc họp đầu tiên của Quốc hội". Chunk này **không chứa tên "Nhà hát lớn"** (tên nằm ở chunk-1), nên cả dense lẫn BM25 đều không kéo nó vào top 5. Generator từ chối đúng (faithfulness = 1) thay vì bịa. |
| 2 | Bài viết về kiến trúc Pháp cổ nhắc đến những công trình tiêu biểu nào? (#8) | dense-only | 1.000 | 0.000 | 0.000 | 0.000 | data | `article_03` chỉ crawl được đoạn về Nhà hát lớn, còn lại là sidebar "tin liên quan". Các cụm "Phủ Chủ tịch", "Nhà thờ lớn", "Cầu Long Biên" **không xuất hiện ở bất kỳ chunk nào trong corpus**, nên golden case này không thể trả lời đủ từ dữ liệu hiện có. |
| 3 | Quy hoạch Thủ đô định hướng bốn trung tâm du lịch chính ở đâu? (#13) | dense-only | 1.000 | 0.450 | 0.000 | 0.200 | retrieval (chunking) | Danh sách 4 trung tâm trong quy hoạch bị chia ra `chunk-435` → `chunk-438` (mỗi trung tâm một đoạn mô tả dài). Top 5 chỉ lấy được chunk-435 (Hồ Hoàn Kiếm – Hồ Tây), nên câu trả lời chỉ nêu 1/4 trung tâm. Hybrid lấy đúng chunk-435 ở hạng 1 (precision 1.00) nhưng recall vẫn bằng 0. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Gắn **contextual header** (title bài và heading gần nhất) vào đầu mỗi chunk trước khi embed và BM25. Sửa splitter để dấu chấm cuối câu ở lại chunk trước, không bị đẩy sang chunk sau. | #7: `article_03::chunk-2` mất chủ ngữ "Nhà hát lớn" và bắt đầu bằng ". Không chỉ…" | Chunk ngắn hoặc chunk tham chiếu ("Đây là…") tìm được bằng tên thực thể; recall #7 từ 0 lên 1 | Index lại, chạy lại 15 câu; kiểm tra #7 có `article_03::chunk-2` trong top 5 và context recall > 0 |
| 2 | **Neighbor-chunk expansion**: với mỗi chunk trong top-k, lấy thêm chunk liền sau cùng tài liệu (giới hạn tổng số ký tự). Hoặc tăng `CHUNK_SIZE` cho văn bản legal dạng liệt kê. | #13: 4 trung tâm nằm ở chunk-435 → 438; #2 dense: món "giò bó mo cau" nằm ở chunk-16, tách khỏi phần mô tả chính | Recall cho câu hỏi dạng liệt kê tăng; answer relevance #13 tăng | So A/B trên cùng 15 câu, chú ý #13 và #2; theo dõi thêm context precision vì context dài hơn |
| 3 | **Crawl lại `article_03`** (đợi nội dung render xong, lọc sidebar), và thêm bước kiểm tra sau crawl: độ dài tối thiểu, có đủ các heading `##` của bài. Nếu trang gốc không có đủ nội dung thì sửa golden case #8. | #8: "Phủ Chủ tịch", "Nhà thờ lớn", "Cầu Long Biên" không có trong corpus; `article_03.md` chỉ 3.8 KB, phần lớn là sidebar | Loại lỗi do dữ liệu; golden set đo đúng năng lực pipeline | Chạy `grep "Phủ Chủ tịch" data/standardized/news/*.md` phải có kết quả; #8 có context recall > 0 |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
|---|---|---:|---:|---|
| Chưa chạy bonus (HyDE, reranker, memory) | — | — | — | Ưu tiên sửa 3 lỗi chunking và dữ liệu ở trên trước. Chênh lệch A/B hiện tại (+0.002) quá nhỏ để đo được tác động của bonus. |
