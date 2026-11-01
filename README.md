# 🚀 Nghiên Cứu & Triển Khai Toàn Diện Kiến Trúc RAG: Từ Advanced Đến Modular RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%20%7C%20LangGraph-orange.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-green.svg)](https://www.trychroma.com/)
[![Embedding](https://img.shields.io/badge/Embedding-BAAI%2Fbge--m3-red.svg)](https://huggingface.co/BAAI/bge-m3)
[![Reranker](https://img.shields.io/badge/Reranker-BAAI%2Fbge--reranker--base-yellow.svg)](https://huggingface.co/BAAI/bge-reranker-base)
[![LLM Support](https://img.shields.io/badge/LLM-Groq%20Cloud%20%7C%20Google%20Gemini-purple.svg)](https://console.groq.com/)

> **Kho lưu trữ nghiên cứu lý thuyết và thực thi thực tế (Theory & Industrial Implementation)** hệ thống **Tạo sinh Tăng cường Truy xuất (Retrieval-Augmented Generation - RAG)** chuẩn mực công nghiệp. 
> Dự án đồng hành cùng bài báo nghiên cứu: ***"Tổng quan kiến trúc RAG"*** — Nhóm tác giả: *Nguyễn Lê Anh Tuấn, Nguyễn Đức Huy, Lê Thành Nam, Hà Minh Châu* (HYCSC / Khoa CNTT / Trường Đại học Xây Dựng Hà Nội).

---

## 📌 Mục Lục
1. [Giới Thiệu Chung & Mục Tiêu](#1-giới-thiệu-chung--mục-tiêu)
2. [Bản Đồ Tiến Hóa Kiến Trúc RAG](#2-bản-đồ-tiến-hóa-kiến-trúc-rag)
3. [Điểm Nổi Bật Kỹ Thuật](#3-điểm-nổi-bật-kỹ-thuật)
4. [Cấu Trúc Thư Mục](#4-cấu-trúc-thư-mục)
5. [Bộ Ba Notebook Thực Thi Chuẩn Công Nghiệp](#5-bộ-ba-notebook-thực-thi-chuẩn-công-nghiệp)
6. [Đánh Giá Định Lượng (LLM-as-a-Judge)](#6-đánh-giá-định-lượng-llm-as-a-judge)
7. [Hướng Dẫn Cài Đặt & Vận Hành](#7-hướng-dẫn-cài-đặt--vận-hành)
8. [Xử Lý Sự Cố Thường Gặp (Troubleshooting)](#8-xử-lý-sự-cố-thường-gặp-troubleshooting)
9. [Tài Liệu Tham Khảo](#9-tài-liệu-tham-khảo)

---

## 1. Giới Thiệu Chung & Mục Tiêu

Dù các mô hình ngôn ngữ lớn (LLMs) có khả năng sinh văn bản xuất sắc nhờ **Bộ nhớ tham số (Parametric Memory)**, chúng vẫn vướng phải các rào cản cố hữu:
- Hiện tượng bịa đặt thông tin (**Hallucination**).
- Tri thức bị đóng băng tại thời điểm kết thúc huấn luyện (**Static Knowledge cutoff**).
- Không thể kiểm chứng hoặc truy xuất nguồn gốc (**Lack of Citations/Traceability**).

Repository này cung cấp giải pháp toàn diện bằng cách triển khai hai thế hệ kiến trúc tối tân:
- **Advanced RAG**: Đường ống tuyến tính 5 giai đoạn được tối ưu hoá nghiêm ngặt về tiền xử lý, định tuyến, tìm kiếm lai và tái xếp hạng.
- **Modular RAG**: Kiến trúc mạng trạng thái động (**LangGraph State Machine**) với năng lực tự phản biện (`[IsRel]`) và vòng lặp tự sửa sai (**Self-Correction Loop**).

---

## 2. Bản Đồ Tiến Hóa Kiến Trúc RAG

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIẾN HOÁ CÁC THẾ HỆ RAG                         │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Naive RAG:                                                          │
│    Indexing ────────► Retrieval (Vector đơn nhất) ────────► Generation │
│    (Điểm mù ngữ cảnh, dễ hallucination, thiếu tính thích ứng)         │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Advanced RAG (Chuỗi 5 giai đoạn tuyến tính chuẩn tắc):              │
│    Small2Big Index ──► Gating/Rewrite ──► Hybrid Search (RRF)          │
│    ──► Cross-Encoder Rerank ──► Generation có trích dẫn                │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Modular RAG (State Machine linh hoạt - LangGraph):                  │
│    Router Node ──► Retrieve & Rerank ──► Grader [IsRel]                │
│    ──► (Nếu irrelevant: Rewrite Query & Retry) ──► Generation          │
└────────────────────────────────────────────────────────────────────────┘
```

| Đặc Tính | Naive RAG | Advanced RAG | Modular RAG |
| :--- | :--- | :--- | :--- |
| **Cấu trúc luồng dữ liệu** | Tuyến tính cố định 3 bước | Tuyến tính mở rộng 5 bước | Mạng trạng thái thích ứng (LangGraph DAG) |
| **Chiến lược Chunking** | Cố định (Fixed-size) | Small-to-Big (Parent-Child) | Small-to-Big đa nguồn |
| **Cơ chế Truy xuất** | Dense Vector đơn nhất | **Hybrid Search**: Dense (BGE-M3) + Sparse (BM25) | Hybrid Search kết hợp định tuyến nguồn |
| **Hợp nhất & Xếp hạng** | Không (lấy Top-K thô) | **RRF (k=60)** + **Cross-Encoder Reranker** | Cross-Encoder Reranker + Threshold Filter |
| **Tự phản biện & Sửa sai** | Không | Cơ bản qua System Prompt | Sâu qua Document Grader & Self-Correction Loop |
| **Thời gian nạp suy luận** | Tái tính toán lại từ đầu | Nạp DB lưu sẵn trong **1 giây** | Nạp DB lưu sẵn trong **1 giây** |

---

## 3. Điểm Nổi Bật Kỹ Thuật

* **Phân Tách Indexing & Inference**: Khâu nặng nhất (đọc PDF, tính vector embedding) chỉ cần chạy **đúng 1 lần duy nhất** (`00_Offline_Indexing.ipynb`). Các phiên suy luận sau đó nạp DB chỉ mất ~1 giây.
* **Chiến Lược Small-to-Big (Parent-Child) Chunking**:
  - *Child Chunks (300 ký tự, overlap 50)*: Tối ưu cho việc so khớp vector embedding và từ khóa BM25 chính xác cao.
  - *Parent Chunks (1200 ký tự, overlap 150)*: Cung cấp đầy đủ ngữ cảnh hoàn chỉnh cho LLM khi sinh câu trả lời, loại bỏ hiện tượng đứt gãy thông tin.
* **Hybrid Search kết hợp Reciprocal Rank Fusion (RRF)**:
  - Vector Dense: `BAAI/bge-m3` đa ngôn ngữ, chuẩn hóa vector.
  - Sparse Keyword: `BM25Okapi` bắt chính xác các thuật ngữ chuyên ngành (Self-RAG, RRF, Gating...).
  - Thuật toán RRF với tham số chuẩn $k=60$:
    $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
* **Cross-Encoder Reranker (`BAAI/bge-reranker-base`)**: Đánh giá toàn diện ma trận chú ý (Cross-Attention) giữa câu hỏi và văn bản, loại bỏ hiện tượng *Lost in the Middle* và chỉ giữ lại Top-3 tài liệu tinh hoa nhất.
* **Định Tuyến & Phản Biện Tự Động (Self-RAG Principles)**:
  - **Gating / Router Node**: Nhận diện câu chào hỏi/xã giao để sinh câu trả lời trực tiếp mà không cần kích hoạt GPU tìm kiếm.
  - **Document Grader (`[IsRel]`)**: Kiểm tra độc lập xem các tài liệu lấy về có thực sự liên quan không.
  - **Self-Correction Edge**: Nếu toàn bộ tài liệu không khớp, tự động kích hoạt **Rewrite Query Node** để đổi góc tiếp cận và tìm kiếm lại (tối đa 2 chu kỳ).
* **Kiểm Soát Ảo Giác & Trích Dẫn**: Generation Prompt ép cấu trúc gạch đầu dòng và trích dẫn số `[1]`, `[2]` tương ứng với nguồn gốc tài liệu.
* **Hỗ Trợ Đa Nhà Cung Cấp LLM**:
  - **Groq Cloud** (Khuyên dùng): Tốc độ ~500 tokens/s, hạn mức miễn phí 14,400 requests/ngày.
  - **Google Gemini API**: Model `gemini-3.1-pro-preview` / `gemini-1.5-pro` tư duy chiều sâu.
  - Tích hợp cơ chế **Auto-Ping Test**: Tự động dò và chọn model đang hoạt động (active) mà không sợ lỗi model deprecated/404.

---

## 4. Cấu Trúc Thư Mục

```
.
├── README.md                                    # Tài liệu giới thiệu & hướng dẫn tổng quan dự án
├── Tổng-quan-RAG.pdf                            # Bài báo khoa học tổng quan kiến trúc RAG (Nhóm nghiên cứu HYCSC)
├── Document/                                    # Thư mục tài liệu lý thuyết & hướng dẫn
│   ├── 1.pdf                                    # Paper: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
│   ├── 2.pdf                                    # Paper: "Retrieval-Augmented Generation for LLMs: A Survey" (Gao et al., 2024)
│   ├── 3.pdf                                    # Paper: "SELF-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023)
│   └── huong_dan_su_dung_bo_ba_notebook_rag.md  # Sổ tay hướng dẫn chi tiết quy trình chạy trên Kaggle & Local
└── Notebook/                                    # Bộ ba Jupyter Notebooks chuẩn công nghiệp
    ├── 00_Offline_Indexing.ipynb                # Khâu 0: Lập chỉ mục ngoại tuyến, xuất ./rag_storage/
    ├── 01_Advanced_RAG_Inference.ipynb          # Khâu 1: Chuỗi Advanced RAG 5 bước, LLM-as-a-Judge & CLI
    ├── 02_Modular_RAG_Inference.ipynb          # Khâu 2: LangGraph State Machine, Adaptive Agent & CLI
    └── RAG.md                                   # Tài liệu kỹ thuật chuyên sâu về các mô-đun và công nghệ RAG
```

---

## 5. Bộ Ba Notebook Thực Thi Chuẩn Công Nghiệp

Hệ thống được thiết kế theo mô hình tách rời chặt chẽ:

```
┌────────────────────────────────────────────────────────────────────────┐
│               00_Offline_Indexing.ipynb (Chạy 1 lần duy nhất)          │
│  - Nạp 3 tài liệu PDF nền tảng (Lewis, Gao Survey, Self-RAG)           │
│  - Phân đoạn Small-to-Big: 1,309 Child Chunks / 292 Parent Chunks      │
│  - Dense Embedding: BAAI/bge-m3                                        │
│  - Sparse Index: BM25Okapi                                             │
│  - Ghi vĩnh viễn: ./rag_storage/ (Chroma SQLite + bm25_index.pkl)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         Dữ liệu nạp sẵn (~1 giây)
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
┌──────────────────────────────────┐┌──────────────────────────────────┐
│ 01_Advanced_RAG_Inference.ipynb  ││ 02_Modular_RAG_Inference.ipynb   │
│ Pipeline Tuyến tính 5 Giai đoạn: ││ LangGraph State Machine:         │
│ 1. Nạp nhanh DB (1 giây)         ││ 1. Nạp nhanh DB (1 giây)         │
│ 2. Pre-Retrieval (Gate/Rewrite)  ││ 2. Router Node (Phân loại)       │
│ 3. Hybrid Search (RRF k=60)      ││ 3. Retrieve & Rerank (Top-3)     │
│ 4. Cross-Encoder Rerank (Top-3)  ││ 4. Document Grader [IsRel]       │
│ 5. Generation (Trích dẫn số)     ││ 5. Self-Correction Loop (Rewrite)│
│ 6. LLM-as-a-Judge Benchmark      ││ 6. Generation & Evaluation       │
│ 7. Interactive Terminal CLI      ││ 7. Interactive Terminal CLI      │
└──────────────────────────────────┘└──────────────────────────────────┘
```

### Bước 1: `Notebook/00_Offline_Indexing.ipynb`
- Quét toàn bộ tài liệu PDF trong thư mục.
- Cắt phân đoạn theo mô hình phân cấp: Parent (1200) $\rightarrow$ Child (300).
- Nhúng vector ngữ nghĩa qua `BAAI/bge-m3` và tạo bảng tra BM25.
- Lưu trữ toàn bộ kết quả vào `./rag_storage/`.

### Bước 2: `Notebook/01_Advanced_RAG_Inference.ipynb`
- Tự động nạp vector store và chỉ mục BM25 trong 1 giây.
- Triển khai hàm `run_advanced_rag(user_query)` gồm 5 chặng liên tiếp:
  1. *Gating*: Lọc câu hỏi cần tìm kiếm hay trả lời trực tiếp.
  2. *Query Rewriter*: Tối ưu câu hỏi thành dạng truy vấn chuyên ngành giàu từ khóa.
  3. *Hybrid Search*: Kết hợp kết quả Dense và Sparse qua giải thuật RRF ($k=60$).
  4. *Rerank*: Cross-Encoder tái định vị Top-3 và tự động ánh xạ ngược về Parent Chunk.
  5. *Generation*: Sinh câu trả lời định dạng chuẩn kèm nguồn trích dẫn.
- Tích hợp bộ đánh giá tự động và CLI tương tác.

### Bước 3: `Notebook/02_Modular_RAG_Inference.ipynb`
- Biên dịch đồ thị trạng thái với **LangGraph**:
  - `decide_route`: Điều hướng sang `direct_gen` nếu là câu hỏi xã giao.
  - `node_grade_docs`: Chấm điểm độ liên quan của từng chunk tài liệu với câu hỏi (`yes` / `no`).
  - `check_relevance`: Nếu không có tài liệu đạt chuẩn, kích hoạt `node_rewrite` để viết lại query và tìm kiếm lại (tối đa 2 lượt thử).
- Tích hợp bộ đánh giá định lượng và CLI tương tác trực tiếp với Agent.

---

## 6. Đánh Giá Định Lượng (LLM-as-a-Judge)

Cả hai notebook suy luận đều tích hợp cơ chế thẩm định chất lượng độc lập **LLM-as-a-Judge** (không phụ thuộc vào các thư viện bên ngoài như Ragas):

| Chỉ Số Đánh Giá | Ý Nghĩa Kỹ Thuật | Mục Tiêu Chuẩn |
| :--- | :--- | :---: |
| **`context_relevance`** | Mức độ liên quan của các đoạn tài liệu truy xuất về so với câu hỏi | **$\ge 0.90$** |
| **`faithfulness`** | Mức độ trung thực của câu trả lời, không bịa đặt dữ kiện ngoài tài liệu | **$\ge 0.95$** |
| **`answer_relevance`** | Câu trả lời có đúng trọng tâm, giải quyết triệt để yêu cầu của câu hỏi | **$\ge 0.90$** |

Hệ thống tự động chấm điểm trên bộ câu hỏi kiểm thử kỹ thuật và xuất kết quả trực tiếp dưới dạng bảng `pandas.DataFrame`.

---

## 7. Hướng Dẫn Cài Đặt & Vận Hành

### 7.1. Chuẩn Bị Môi Trường Cục Bộ (Local Machine)

1. **Yêu cầu hệ thống**:
   - Python 3.10 trở lên.
   - GPU NVIDIA (khuyên dùng để chạy BGE-M3 và Cross-Encoder mượt mà; CPU vẫn chạy được).

2. **Cài đặt thư viện**:
```bash
pip install -q -U \
    langchain \
    langchain-community \
    langchain-groq \
    langchain-google-genai \
    langgraph \
    sentence-transformers \
    rank-bm25 \
    pypdf \
    chromadb \
    pandas \
    datasets
```

3. **Cấu hình API Key**:
Cung cấp ít nhất 1 trong 2 khóa API (khuyên dùng **Groq** vì tốc độ cao và hạn mức rộng):
```bash
# Khuyên dùng: Groq Cloud (Miễn phí 14,400 requests/ngày)
export GROQ_API_KEY="gsk_..."

# Hoặc: Google Gemini API
export GEMINI_API_KEY="AIzaSy..."
```

4. **Thực thi**:
- Mở và chạy toàn bộ `Notebook/00_Offline_Indexing.ipynb` để sinh thư mục `./rag_storage/`.
- Mở `Notebook/01_Advanced_RAG_Inference.ipynb` hoặc `Notebook/02_Modular_RAG_Inference.ipynb` để hỏi đáp và xem benchmark.

---

### 7.2. Vận Hành Trên Kaggle (Tối Ưu & Miễn Phí)

1. **Cấu hình phần cứng (Notebook Settings)**:
   - **File `00`**: Chọn **GPU T4 x 2** (hoặc GPU T4 x 1) để nhúng vector nhanh.
   - **File `01` & `02`**: Có thể chọn **CPU** (để tiết kiệm quota GPU) hoặc GPU T4.
   - **Internet**: Gạt sang **Internet On** (bắt buộc).
2. **Kaggle Secrets**:
   - Chọn menu **Add-ons** ➔ **Secrets**.
   - Thêm `GROQ_API_KEY` (hoặc `GEMINI_API_KEY`).
3. **Liên kết Output**:
   - Chạy xong Notebook `00`, bấm **Save Version** (Save & Run All).
   - Tại Notebook `01` hoặc `02`, chọn **Add Input** ➔ **Notebooks** ➔ Chọn output của Notebook `00`. Code đã tích hợp sẵn cơ chế tự động phát hiện đường dẫn và sao chép quyền ghi sang `/kaggle/working/rag_storage`.

---

## 8. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Vấn Đề | Nguyên Nhân | Giải Pháp Đã Tích Hợp |
| :--- | :--- | :--- |
| `attempt to write a readonly database (code: 8)` | SQLite trên Kaggle bị khóa quyền ghi do dữ liệu nằm trong `/kaggle/input/`. | Notebook `01` và `02` đã tích hợp sẵn lệnh `shutil.copytree` tự động chuyển sang `/kaggle/working/rag_storage`. |
| `NotFoundError: Model does not exist (404)` | Gọi sai tên model API của Groq hoặc Gemini. | Code tích hợp cơ chế **Ping Test**: Tự động test 1 token để kích hoạt đúng model đang active trong tài khoản. |
| `ResourceExhausted / 429 Quota Exceeded` | Vượt hạn mức 20 req/ngày của Gemini Free Tier. | Đổi sang dùng **Groq API Key** (`GROQ_API_KEY`) với hạn mức 14,400 req/ngày miễn phí hoàn toàn. |
| Lặp từ vô tận (`đề đề đề...`) | `temperature = 0.0` trên các mô hình open-weight gây kẹt greedy decoding. | Cấu hình mặc định tối ưu: `temperature = 0.2`, `max_tokens = 1024`. |

---

## 9. Tài Liệu Tham Khảo

1. **Patrick Lewis et al. (2020)**. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) *(đính kèm tại `Document/1.pdf`)*.
2. **Yunfan Gao et al. (2024)**. *Retrieval-Augmented Generation for Large Language Models: A Survey*. [arXiv:2312.10997](https://arxiv.org/abs/2312.10997) *(đính kèm tại `Document/2.pdf`)*.
3. **Akari Asai et al. (2023)**. *SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) *(đính kèm tại `Document/3.pdf`)*.
4. **Nguyễn Lê Anh Tuấn, Nguyễn Đức Huy, Lê Thành Nam, Hà Minh Châu**. *Tổng quan kiến trúc RAG*. HYCSC / Khoa CNTT / Trường Đại học Xây Dựng Hà Nội *(đính kèm tại `Tổng-quan-RAG.pdf`)*.

---
*Dự án được xây dựng và duy trì bởi nhóm nghiên cứu HYCSC - Khoa CNTT, Trường Đại học Xây Dựng Hà Nội.*

# RAG-repository
