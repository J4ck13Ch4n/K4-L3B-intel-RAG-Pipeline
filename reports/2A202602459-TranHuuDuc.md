# Individual contribution report

## Thông tin

- Họ và tên: Trần Hữu Đức
- Mã học viên: 2A202602459
- Nhóm: intel
- Repository/branch: K4-L3B-intel-RAG-Pipeline

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Môi trường WSL và indexing (Task 4) | Chuyển provider sang OpenAI; sửa lỗi Gemini client bị GC đóng giữa request; thêm retry khi gặp 429 và index tiếp từ chỗ dừng; đặt `CHROMA_DIR` trong filesystem Linux (SQLite hỏng khi ghi qua `/mnt/d`); tách collection theo embedding model để không trộn vector 768 và 1536 chiều | `src/task4_chunking_indexing.py`, `.env.example` | Done |
| Hiệu chỉnh threshold fallback (Task 9) | Đo best dense cosine trên 15 câu golden và 5 câu ngoài domain, chọn `SCORE_THRESHOLD=0.45`, đưa giá trị này vào code và `.env.example` | `src/task9_retrieval_pipeline.py`, `.env.example` | Done |
| PageIndex fallback (Task 8) | Viết lại theo API thật của `pageindex==0.2.8` (`submit_document` / `submit_query` / `get_retrieval`); bản cũ gọi `client.index()` và `get_document_structure()` vốn không tồn tại. Upload 3 PDF, parse `relevant_contents` dạng list lồng list, có timeout và query song song | `src/task8_pageindex_vectorless.py` | Done, đã chạy với API thật |
| Generation contract (Task 10) | Sửa `sources` bị trả theo thứ tự đã reorder (vi phạm yêu cầu sort theo score); nhãn `[S<n>]` luôn map về `sources[n-1]`; khi từ chối trả `sources=[]` | `src/task10_generation.py` | Done |
| Evaluation | Phân tích `raw_results.json` của lượt A/B nhóm (gpt-4o): đối chiếu câu trả lời với chunk thật để tìm root cause 3 câu tệ nhất, viết lại A/B comparison, worst performers và recommendations. Sửa `evaluate.py`: chấm context theo thứ tự retrieval gốc, latency chỉ đo retrieval + generation, retry khi judge timeout, hỗ trợ evaluator OpenAI | `src/evaluate.py`, `group_project/evaluation/RESULT.md` | Done (chưa chạy lại A/B với bản sửa) |
| Demo scripts | `__main__` của Task 5/6/7/9 chạy query thật; Task 9 in best dense cosine so với threshold | `src/task5`–`task9` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Đặt threshold fallback = 0.45 trên dense cosine gốc, không dùng RRF score.
   **Lý do/evidence:** Với `text-embedding-3-small`, best cosine của 15 câu golden nằm trong 0.471–0.737, còn của 5 câu ngoài domain (nấu ăn, bitcoin, bóng đá, cài Python, thời tiết Paris) nằm trong 0.254–0.421. Ngưỡng 0.45 tách hai nhóm; giá trị mặc định cũ 0.3 thì không chặn được câu nào ngoài domain.
   **Trade-off:** Biên an toàn hẹp (0.021 so với câu golden thấp nhất là "Ba Vì–Suối Hai"), nên câu hỏi hợp lệ nhưng diễn đạt lạ có thể rơi vào fallback chậm (khoảng 40 giây). Threshold chỉ đúng cho embedding model này; đổi model thì phải hiệu chỉnh lại.

2. **Quyết định:** Viết lại Task 8 dựa trên response thật của PageIndex thay vì tài liệu SDK.
   **Lý do/evidence:** Bản cũ gọi hàm SDK không tồn tại, nhưng lỗi bị `retrieve()` nuốt mất nên fallback chưa bao giờ chạy. Gọi thử API cho thấy node dùng field `id` (không phải `node_id`) và `relevant_contents` là list lồng list.
   **Trade-off:** Mỗi lần fallback phải truy vấn cả 3 tài liệu (khoảng 23 giây). Endpoint retrieval đã bị PageIndex đánh dấu deprecated, nên về lâu dài cần chuyển sang chat API.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest -q` (20/20 pass). Query in-domain "Nhà hát lớn Hà Nội được xây dựng năm nào?" cho dense 0.759 → hybrid → "1901–1911 [S1]". Query ngoài domain "Tỷ giá bitcoin hôm nay?" cho dense 0.397 → PageIndex → safe refusal. Query legal ép fallback cho `retrieval_source=pageindex`, trả lời "khoảng 8% GRDP [S1]". Kiểm tra chatbot bằng `streamlit.testing.AppTest` (2 lượt hỏi, không exception).
- Kết quả trước/sau nếu có:
  - Fallback: trước, câu "Tỷ giá bitcoin hôm nay?" (dense 0.397) vượt threshold 0.3 nên không kích hoạt fallback, và PageIndex nếu được gọi cũng lỗi ngầm. Sau: 0.397 < 0.45 → PageIndex chạy thật → safe refusal (`retrieval_source=none`, khoảng 44 giây end-to-end).
  - Contract Task 10: trước, `sources` có score 0.0164 → 0.0159 → 0.0154 → 0.0156 → 0.0161 (không sort, `validate_generation_result` fail). Sau: sort giảm dần, `[S1]` ứng với `sources[0]`.
  - A/B (lượt gpt-4o của nhóm, tôi phân tích): hybrid + RRF 0.7234 so với dense-only 0.7213 (+0.0021, xét từng câu là 5 thắng – 5 thua – 5 hoà). Context recall bằng nhau (0.80) vì cả hai config cùng trượt #7, #8, #13: hai câu do chunking, một câu do dữ liệu crawl thiếu.
- Lỗi đã phát hiện và cách xử lý: (1) `sources` không sort theo score → `validate_generation_result` fail → giữ thứ tự retrieval và gán nhãn citation theo vị trí gốc. (2) `chroma_db` hỏng trên `/mnt/d` (`FileExistsError`) → chuyển sang `~/.cache/hanoi-rag`. (3) Quota free tier Gemini 100 text/phút → retry theo `retryDelay` rồi chuyển hẳn sang OpenAI. (4) Lượt chạy lại evaluation dừng ngay câu đầu vì một lần `APITimeoutError` (instructor chỉ thử 1 lần) → thêm `timeout=120, max_retries=5` và chấm lại tối đa 3 lần mỗi câu.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: fallback PageIndex chậm (câu ngoài domain mất khoảng 40 giây end-to-end) và phụ thuộc endpoint đã deprecated; `doc_id` gắn với tài khoản PageIndex của tôi nên thành viên khác phải tự upload lại.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: gắn contextual header (title và heading) vào mỗi chunk để sửa lỗi retrieval ở câu #7 (chunk chứa đáp án không có tên "Nhà hát lớn"), rồi chạy lại A/B trên đúng 15 câu golden.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Trần Hữu Đức
