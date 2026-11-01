# HƯỚNG DẪN THỰC THI TOÀN DIỆN: BỘ BA NOTEBOOK RAG (00, 01, 02)

Tài liệu này hướng dẫn chi tiết cách vận hành bộ 3 Jupyter Notebooks triển khai các kiến trúc **Advanced RAG** và **Modular RAG** từ khâu lập chỉ mục ngoại tuyến (offline indexing) đến suy luận và đánh giá định lượng (online inference & evaluation).

---

## 1. Bản Đồ Phân Phối Kiến Trúc & Nhiệm Vụ

Hệ thống được tổ chức thành 3 module độc lập theo mô hình Production chuẩn công nghiệp:

```
┌────────────────────────────────────────────────────────────────────────┐
│               00_Offline_Indexing.ipynb (Chạy 1 lần)                   │
│  - Đọc toàn bộ PDF (Small-to-Big Chunking: 1200 / 300)                 │
│  - Dense Embedding: BAAI/bge-m3 (GPU T4)                               │
│  - Sparse Keyword Index: BM25Okapi                                     │
│  - Xuất dữ liệu vĩnh viễn: ./rag_storage/                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ Dữ liệu đã lập chỉ mục (ChromaDB + Pickle)
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│ 01_Advanced_RAG_Inference.ipynb │   │  02_Modular_RAG_Inference.ipynb │
│ Pipeline Tuyến tính 5 Giai đoạn:│   │  LangGraph State Machine:       │
│ 1. Nạp nhanh DB (1 giây)        │   │ 1. Nạp nhanh DB (1 giây)        │
│ 2. Pre-Retrieval (Gate/Rewrite) │   │ 2. Router Node (Phân loại)      │
│ 3. Hybrid Search (RRF k=60)     │   │ 3. Retrieve & Rerank            │
│ 4. Cross-Encoder Rerank (Top-3) │   │ 4. Grader [IsRel] (Phản biện)   │
│ 5. Generation (Trích dẫn)       │   │ 5. Self-Correction Loop         │
│ 6. Benchmark LLM-as-a-Judge     │   │ 6. Generation & Evaluation      │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 2. Chuẩn Bị Tài Nguyên & API Keys

### 2.1. API Keys (Chọn 1 trong 2 nhà cung cấp)

| Nhà cung cấp | Khuyến nghị | Hạn mức Free Tier | Ưu điểm | Link lấy Key |
| :--- | :---: | :---: | :--- | :--- |
| **Groq Cloud** | ⭐⭐⭐ **Khuyên dùng** | **14,400 requests / ngày** | Tốc độ siêu thanh (~500 t/s), không lo hết quota khi benchmark | [console.groq.com/keys](https://console.groq.com/keys) |
| **Google Gemini** | ⭐⭐ Lựa chọn thay thế | Giới hạn 20 req/ngày (với key mới tạo chưa billing) | Mô hình `gemini-3.1-pro-preview` tư duy sâu | [aistudio.google.com](https://aistudio.google.com/) |

### 2.2. Thiết lập trên Kaggle (Bảo mật & Tối ưu phần cứng)
1. Mở Kaggle Notebook, nhìn sang thanh công cụ bên phải (**Settings**):
   - **Accelerator**: 
     - Với file `00`: Chọn **GPU T4 x 2** (hoặc GPU T4 x 1) để tính embedding nhanh.
     - Với file `01` và `02`: Có thể chọn **CPU** (để tiết kiệm giờ GPU) hoặc **GPU T4** đều được.
   - **Internet**: Gạt sang **Internet On** (bắt buộc để tải model HuggingFace và gọi API).
2. Lưu API Key vào Kaggle Secrets:
   - Vào menu trên cùng: **Add-ons** ➔ **Secrets**.
   - Thêm nhãn `GROQ_API_KEY` (hoặc `GEMINI_API_KEY`) và dán giá trị key tương ứng.

---

## 3. Quy Trình Vận Hành Chi Tiết Từng Notebook

### BƯỚC 1: Lập chỉ mục với `00_Offline_Indexing.ipynb` (Chỉ chạy 1 lần duy nhất)

* **Mục tiêu**: Nạp các file PDF paper RAG (`1.pdf`, `2.pdf`, `3.pdf`), cắt nhỏ theo chiến lược Small-to-Big, tính vector và lưu trữ vĩnh viễn xuống ổ đĩa.
* **Các bước thực hiện**:
  1. Đặt 3 file PDF vào cùng thư mục làm việc (hoặc upload thành Kaggle Dataset).
  2. Mở file `00_Offline_Indexing.ipynb`.
  3. Bấm **Run All**.
  4. **Kết quả đầu ra**:
     - Cắt thành **292 Parent Chunks** và **1,309 Child Chunks**.
     - Khởi tạo thư mục `./rag_storage/` chứa `chroma.sqlite3` và `bm25_index.pkl`.
  5. **Lưu Output trên Kaggle**:
     - Bấm **Save Version** ➔ Chọn **Save & Run All (Commit)**.
     - Khi notebook chạy xong, thư mục output sẽ trở thành một Dataset có thể liên kết vào các notebook khác.

---

### BƯỚC 2: Truy vấn & Đánh giá với `01_Advanced_RAG_Inference.ipynb`

* **Mục tiêu**: Triển khai đường ống Advanced RAG 5 bước chuẩn mực. Nạp DB trong 1 giây, thực hiện đánh giá định lượng LLM-as-a-Judge và cung cấp giao diện hỏi đáp terminal.
* **Cách liên kết dữ liệu trên Kaggle**:
  - Tại giao diện notebook `01`, ở khung **Input** góc trên bên phải, bấm **Add Input**.
  - Chọn thẻ **Notebooks** ➔ Tìm và thêm output của notebook `00_Offline_Indexing` vừa lưu.
* **Các bước thực thi**:
  1. Bấm **Run All** (hoặc chạy tuần tự từng cell).
  2. **Cell 2 (Khởi động hệ thống)**:
     - Tự động phát hiện cơ sở dữ liệu từ `/kaggle/input/.../rag_storage`.
     - Tự động copy sang `/kaggle/working/rag_storage` để mở khóa quyền ghi cho SQLite.
     - Tự động gửi **Ping Test** chọn đúng model ngôn ngữ đang hoạt động trên Groq/Gemini.
     - Nạp Reranker `BAAI/bge-reranker-base`.
  3. **Cell 4 (Đánh giá định lượng Benchmark)**:
     - Chạy bộ câu hỏi kiểm thử qua cơ chế **LLM-as-a-Judge**.
     - Xuất ra bảng DataFrame hiển thị 3 chỉ số vàng:
       - `context_relevance` ($\ge 0.90$)
       - `faithfulness` ($\ge 0.95$)
       - `answer_relevance` ($\ge 0.90$)
  4. **Cell 5 (Interactive CLI)**:
     - Xuất hiện dấu nhắc: `👉 Bạn muốn hỏi gì:`.
     - Bạn có thể gõ bất kỳ câu hỏi kỹ thuật nào liên quan đến 3 tài liệu RAG. Hệ thống sẽ trả lời phân mục gạch đầu dòng kèm trích dẫn số `[1]`, `[2]`.
     - Gõ `exit` hoặc `quit` để kết thúc phiên.

---

### BƯỚC 3: Thử nghiệm Agent tự sửa sai với `02_Modular_RAG_Inference.ipynb`

* **Mục tiêu**: Triển khai hệ thống Modular RAG dựa trên đồ thị trạng thái **LangGraph**, trang bị năng lực tự phản biện (Self-Reflection) và tự sửa sai (Self-Correction).
* **Các cơ chế động được kích hoạt**:
  - **Router Node**: Nếu nhận câu chào hỏi xã giao (ví dụ: *"Chào bạn, hôm nay thế nào?"*), hệ thống trả lời trực tiếp mà không kích hoạt GPU tìm kiếm.
  - **Document Grader Node (`[IsRel]`)**: Lọc bỏ các chunk không liên quan câu hỏi.
  - **Self-Correction Edge**: Nếu toàn bộ tài liệu tìm về đều không đạt `[IsRel]`, hệ thống tự động bẻ luồng sang node **Rewrite Query** để viết lại truy vấn và tìm kiếm lại (tối đa 2 chu kỳ).
  - **Generation Node**: Trả lời kèm trích dẫn kiểm soát ảo giác.
* **Các bước thực thi**:
  1. Thêm Input từ notebook `00` (tương tự như bước 2).
  2. Bấm **Run All**.
  3. Quan sát nhật ký (logs) luồng đi của Agent tại các node: `[AGENT: RETRIEVE]`, `[AGENT: GRADE DOCS]`, `[AGENT: GENERATE]`.
  4. Sử dụng khung hỏi đáp ở cell cuối cùng để tương tác đa lượt với Agent.

---

## 4. Bảng Tra Cứu Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Hiện tượng / Thông báo lỗi | Nguyên nhân | Cách xử lý |
| :--- | :--- | :--- |
| `attempt to write a readonly database (code: 8)` | Kaggle `/kaggle/input/` bị khóa quyền ghi khiến ChromaDB SQLite không tạo được file lock. | Đã tích hợp sẵn lệnh `shutil.copytree` tự động chuyển sang `/kaggle/working/rag_storage` trước khi load. |
| `NotFoundError: Model does not exist (404)` | Gọi sai tên model API của Groq hoặc Gemini. | Code đã trang bị cơ chế **Ping Test tự động**: Thử nghiệm gọi 1 token trước, model nào mở quyền thì kích hoạt model đó. |
| `ResourceExhausted / 429 Quota Exceeded` | Hết quota 20 req/ngày của Gemini Free Tier. | Đổi sang dùng **Groq API Key** (`GROQ_API_KEY`) để nhận 14,400 req/ngày miễn phí hoàn toàn. |
| Điểm `context_relevance` hoặc `faithfulness` bị `0.0` | Câu hỏi bị Gating phân loại nhầm là `NO_SEARCH`, LLM trả lời bằng trí nhớ nội tại không dùng tài liệu. | Đảm bảo câu hỏi có chứa từ khóa chuyên môn (Self-RAG, Dense retrieval, BM25, Reranker...) để ép hệ thống kích hoạt khâu Retrieval. |
| Lặp từ vô tận (`đề đề đề đề đề...`) | `temperature = 0.0` trên một số LLM mã nguồn mở gây kẹt xác suất greedy decoding. | Đã cấu hình cố định `temperature = 0.2` và `max_tokens = 1024` trong prompt generation. |

---

## 5. Kết Luận Kiến Trúc
Bộ ba notebook này hoàn thành trọn vẹn chu trình kỹ thuật từ **Nghiên cứu lý thuyết** (trong tài liệu `t_i_li_u_k_thu_t_to_n_di_n_ki_n_tr_c_advanced_modular_rag.md`) đến **Triển khai thực tế**:
- Tách bạch khâu tính toán nặng (Indexing) và khâu phục vụ người dùng (Inference).
- Tiết kiệm 95% thời gian khởi động lại mô hình.
- Đạt điểm số đánh giá chuẩn mực công nghiệp ($\ge 0.90$) trên các bài báo khoa học gốc.

