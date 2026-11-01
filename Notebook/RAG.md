# TÀI LIỆU KỸ THUẬT TOÀN DIỆN: TỪ ADVANCED RAG ĐẾN MODULAR RAG

---

## 1. Bản đồ Tiến hóa Kiến trúc: Naive $\rightarrow$ Advanced $\rightarrow$ Modular

Sự phát triển của hệ thống Retrieval-Augmented Generation (RAG) chia làm 3 thế hệ kiến trúc rõ rệt:

```
[ Thế hệ 1: Naive RAG ]
   Pipeline cố định 3 bước: Indexing ──► Retrieval ──► Generation
   (Nhiều điểm mù, dễ hallucination, thiếu tính linh hoạt)
            │
            ▼
[ Thế hệ 2: Advanced RAG ]
   Pipeline mở rộng 5 bước chuẩn tắc: 
   Indexing ──► Pre-Retrieval ──► Retrieval ──► Post-Retrieval ──► Generation
   (Bổ sung tối ưu truy vấn, Reranking, Self-Reflection)
            │
            ▼
[ Thế hệ 3: Modular RAG ]
   Kiến trúc Hướng Dịch vụ / Mô-đun động (Microservice-like / Plug-and-Play)
   (Phi tuyến tính, tự định tuyến, tái cấu trúc luồng theo ngữ cảnh và độ tin cậy)
```

| Đặc tính | Naive RAG | Advanced RAG | Modular RAG |
| :--- | :--- | :--- | :--- |
| **Cấu trúc luồng dữ liệu** | Tuyến tính cố định 3 bước | Tuyến tính mở rộng 5 bước | Mạng lưới linh hoạt (DAG / State Machine) |
| **Khả năng tương thích nguồn** | Vector Database đơn nhất | Vector DB + Từ khóa (BM25) | Đa nguồn: Vector, SQL, Graph, Web, Bộ nhớ |
| **Cơ chế điều phối** | Gọi 1 lần duy nhất ($Single\text{-}shot$) | Gọi 1 lần có lọc và Rerank | Động: Thích ứng, Lặp lại hoặc Đệ quy |
| **Khả năng tự phản biện** | Không | Cơ bản (ở prompt cuối) | Sâu rộng qua từng nút (Reflection Tokens) |
| **Độ trễ trung bình** | Thấp ($< 1s$) | Trung bình ($1s - 2s$) | Biến thiên theo độ khó câu hỏi ($0.5s - 5s$) |

---

## 2. Kiến trúc Advanced RAG (Chuỗi 5 Giai đoạn)

Advanced RAG thiết lập một quy trình 5 bước nghiêm ngặt nhằm khắc phục các điểm nghẽn về ngữ cảnh và độ chính xác:

```
┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────┐
│  1. Indexing    │ ───►  │    2. Pre-Retrieval    │ ───►  │  3. Retrieval   │ ───►  │   4. Post-Retrieval    │ ───►  │  5. Generation  │
│ Cắt & Lập chỉ   │       │ Tối ưu & Định tuyến    │       │ Tìm kiếm kết    │       │ Xếp hạng (Rerank) &    │       │ LLM Tạo sinh &  │
│ mục nâng cao    │       │ truy vấn người dùng    │       │ hợp (Hybrid)    │       │ Tinh gọn ngữ cảnh      │       │ Tự phản biện    │
└─────────────────┘       └────────────────────────┘       └─────────────────┘       └────────────────────────┘       └─────────────────┘
```

1. **Indexing:** Triển khai **Small2Big (Parent-Child) Chunking** kết hợp Metadata đa tầng để bảo toàn cả độ chính xác vector lẫn ngữ cảnh nguyên bản.
2. **Pre-Retrieval:** Tối ưu hóa truy vấn thông qua **Query Rewriting** và **Adaptive Gating** nhằm loại trừ truy vấn rác.
3. **Retrieval:** Thực hiện **Hybrid Search** kết hợp giữa Dense Vector Search và Sparse Search (BM25), dung hòa bằng giải thuật **Reciprocal Rank Fusion (RRF)**:
   $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
4. **Post-Retrieval:** Chấm điểm lại qua **Cross-Encoder Reranker**, chọn lấy $Top\text{-}3 \rightarrow Top\text{-}5$ tài liệu tinh hoa nhất, triệt tiêu hiện tượng *Lost in the Middle*.
5. **Generation:** Ứng dụng **In-Context Prompting** đi kèm **Citation Control** và các ràng buộc phản biện (`IsRel`, `IsSup`, `IsUse`).

---

## 3. Modular RAG (RAG Mô-đun)

Trong Modular RAG, hệ thống phá bỏ giới hạn của đường ống tuần tự. Mỗi chức năng được đóng gói thành một mô-đun độc lập có thể được kích hoạt, lặp lại, đảo thứ tự hoặc bỏ qua dựa trên ngữ cảnh thực tế.

```
                   ┌────────────────────────────────────────┐
                   │        Orchestrator / Router           │
                   │   (LangGraph / LlamaIndex Workflow)   │
                   └──────────────────┬─────────────────────┘
                                      │
         ┌───────────────┬────────────┴───┬────────────────┬──────────────┐
         ▼               ▼                ▼                ▼              ▼
  ┌─────────────┐ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ ┌───────────┐
  │   Rewrite   │ │ RAG-Fusion  │  │   Search    │  │   Rerank    │ │  Memory   │
  │ Transform   │ │ Multi-Query │  │ Multi-Source│  │Cross-Encoder│ │Short/Long │
  └─────────────┘ └─────────────┘  └─────────────┘  └─────────────┘ └───────────┘
         ▲                                │                ▲              │
         │                                ▼                │              ▼
  ┌─────────────┐                  ┌─────────────┐  ┌─────────────┐ ┌───────────┐
  │ TaskAdapter │                  │ Vector / DB │  │   Predict   │ │Generation │
  │ Classification                 │ SQL / Graph │  │  GenRead    │ │& Critique │
  └─────────────┘                  └─────────────┘  └─────────────┘ └───────────┘
```

### 3.1. Chi tiết 8 Mô-đun cốt lõi & Đề xuất Mô hình / Công nghệ

#### 1. Search Module (Mô-đun Tìm kiếm Đa nguồn)
* **Chức năng:** Không chỉ dừng lại ở vector database thuần túy, mô-đun này kết nối linh hoạt tới nhiều kho tri thức khác nhau (Web Search, SQL Database, Knowledge Graph, Document Store).
* **Mô hình & Công cụ đề xuất:**
  * *Vector Database:* **Qdrant**, **Milvus**, hoặc **pgvector** (PostgreSQL).
  * *Dense Embedding:* `BAAI/bge-m3` (hỗ trợ đa ngôn ngữ, độ dài token tới 8192, xuất đồng thời dense + sparse vector), `text-embedding-3-large` (OpenAI).
  * *Knowledge Graph Engine:* **Neo4j** kết hợp **Microsoft GraphRAG** (trích xuất thực thể, phân cụm cộng đồng Leiden).
  * *Text-to-SQL Engine:* Mô hình chuyên biệt `defog/sqlcoder-70b-alpha` hoặc `Vanna.ai`.
  * *Live Web Engine:* **Tavily Search API** hoặc **Exa.ai**.

#### 2. Rewrite / Transformation Module (Mô-đun Biến đổi Truy vấn)
* **Chức năng:** Làm sạch câu hỏi, xử lý đại từ chỉ định, viết lại truy vấn rõ nghĩa hơn hoặc chuyển đổi định dạng câu hỏi (HyDE - sinh tài liệu giả định).
* **Mô hình & Công cụ đề xuất:**
  * *Mô hình tối ưu chi phí/tốc độ:* `Qwen2.5-7B-Instruct`, `Llama-3.1-8B-Instruct` hoặc `GPT-4o-mini`.
  * *Kỹ thuật:* **HyDE** (*Hypothetical Document Embeddings* - cho phép mô hình giả định một đoạn nội dung khả dĩ rồi mới embed), **Step-Back Prompting** (lùi lại hỏi một khái niệm tổng quát hơn).

#### 3. RAG-Fusion / Multi-Query Module (Mô-đun Mở rộng & Hợp nhất)
* **Chức năng:** Tách câu hỏi của người dùng thành $N$ câu hỏi phụ mang các sắc thái tiếp cận khác nhau ($Q \rightarrow \{Q_1, Q_2, Q_3\}$), thực thi tìm kiếm song song và dùng thuật toán xếp hạng hợp nhất để gom kết quả.
* **Mô hình & Công cụ đề xuất:**
  * *LLM Generator:* `Llama-3.1-8B-Instruct` với template prompt tạo 3-5 biến thể truy vấn.
  * *Fusion Algorithm:* **Reciprocal Rank Fusion (RRF)** hoặc **DBSCAN Clustering** gom cụm các chunk trùng lặp.

#### 4. Routing Module (Mô-đun Định tuyến Thông minh)
* **Chức năng:** "Cảnh sát giao thông" của hệ thống. Nhận diện tính chất câu hỏi để điều hướng sang kho dữ liệu tương ứng:
  * Câu hỏi phân tích số liệu $\rightarrow$ SQL Module.
  * Câu hỏi quan hệ đa tầng $\rightarrow$ GraphRAG Module.
  * Câu hỏi chính sách, tài liệu $\rightarrow$ Vector DB Module.
  * Câu hỏi hội thoại thường thức $\rightarrow$ Direct LLM (không tra cứu).
* **Mô hình & Công cụ đề xuất:**
  * *Fast Semantic Router (Độ trễ $< 15ms$):* Thư viện **Semantic Router** (Aurelio AI) chạy mô hình embedding siêu nhẹ (`all-MiniLM-L6-v2` hoặc `bge-small-en-v1.5`).
  * *LLM-based Router:* Gọi Function Calling / Structured Output của `GPT-4o-mini` hoặc `Llama-3.1-8B-Instruct`.

#### 5. Memory Module (Mô-đun Bộ nhớ Ngữ cảnh)
* **Chức năng:** Duy trì bộ nhớ ngắn hạn (lịch sử hội thoại vòng lặp) và bộ nhớ dài hạn (thông tin cá nhân hóa, tri thức người dùng đã xác nhận từ các phiên trước).
* **Mô hình & Công cụ đề xuất:**
  * *Giải pháp chuyên sâu:* **Mem0** (bộ nhớ cá nhân hóa theo thời gian thực), **Letta / MemGPT** (quản lý bộ nhớ dạng OS paging), **Zep** (trích xuất facts theo đồ thị tự động).

#### 6. Rerank Module (Mô-đun Tái định vị Thứ hạng)
* **Chức năng:** So sánh toàn diện cặp câu hỏi và tài liệu (Query - Document cross-attention), gạt bỏ nhiễu và định vị lại các đoạn văn bản quan trọng nhất lên đầu.
* **Mô hình & Công cụ đề xuất:**
  * *Open-source SOTA:* `BAAI/bge-reranker-v2-m3` (hỗ trợ đa ngôn ngữ vượt trội), `jinaai/jina-reranker-v2-base-multilingual`.
  * *Commercial API:* **Cohere Rerank v3.5**.

#### 7. Predict / Generate Module (Mô-đun Tri thức Tiềm ẩn)
* **Chức năng:** Tận dụng tri thức đã được huấn luyện sẵn trong trọng số của LLM (Parametric Memory) để sinh trước một bản tóm tắt hoặc dự đoán câu trả lời (*GenRead / Recite-Read*) trước khi hoặc song song với việc tìm kiếm bên ngoài.
* **Mô hình & Công cụ đề xuất:**
  * Các mô hình LLM có năng lực reasoning mạnh: `Qwen2.5-72B-Instruct`, `Llama-3.1-70B-Instruct`, `Claude 3.5 Sonnet`.

#### 8. Task Adapter Module (Mô-đun Thích ứng Tác vụ)
* **Chức năng:** Tự động điều biến cấu trúc đầu ra phù hợp với yêu cầu nghiệp vụ chuyên biệt (JSON Schema, tóm tắt điều hành, phân loại rủi ro, bảng đối soát).
* **Mô hình & Công cụ đề xuất:**
  * **DSPy** (biên dịch prompt tự động và tối ưu hóa trọng số few-shot dựa trên metrics).
  * **Instructor** hoặc **Outlines** (ép kiểu dữ liệu đầu ra chuẩn Pydantic / JSON Schema nghiêm ngặt).

---

## 4. Các Cơ chế Vận hành Động (Augmentation Patterns)

Một câu hỏi quan trọng trong Modular RAG: **"Hệ thống dựa trên điều gì để quyết định gọi mô-đun nào và gọi bao nhiêu lần?"** Câu trả lời nằm ở 3 cơ chế điều phối động sau:

```
      [1. Adaptive Pattern]              [2. Iterative Pattern]              [3. Recursive Pattern]
   Kích hoạt theo Độ Tin cậy          Vòng lặp Luân phiên Liên tục        Bẻ nhỏ Bài toán Suy luận Cây
  ┌───────────────────────────┐      ┌───────────────────────────┐       ┌───────────────────────────┐
  │ Query ──► Confidence Test │      │ Step 1: Search            │       │ Query: Multi-hop QA       │
  │   ├─ High ──► Direct Gen  │      │ Step 2: Draft Gen         │       │   ├── Q1 ──► Retrieve d1  │
  │   └─ Low  ──► Run Search  │      │ Step 3: Need more? ──Loop │       │   └── Q2(d1) ──► Retrieve│
  └───────────────────────────┘      └───────────────────────────┘       └───────────────────────────┘
```

### 4.1. Adaptive Pattern (Thích ứng theo Nhu cầu & Tự phản biện)

* **Nguyên lý kích hoạt:** Dựa trên **Độ tin cậy nội tại (Confidence Level)** của LLM hoặc **Token Phản biện (Reflection Tokens)**.
* **Cơ chế hoạt động:**
  1. **Self-RAG Method:**
     * Mô hình được huấn luyện để sinh ra các token kiểm soát đặc biệt:
       * `[Retrieve]`: Đánh giá câu hỏi có chứa thông tin cần tra cứu không. Nếu `[Retrieve] = No` (ví dụ câu hỏi: *"Hãy viết một bài thơ về mùa thu"*), hệ thống bỏ qua hoàn toàn mô-đun Search.
       * `[IsRel]`: Đánh giá tài liệu lấy về có liên quan hay không. Nếu `[IsRel] = Irrelevant`, hệ thống hủy kết quả và kích hoạt tìm kiếm lại với câu truy vấn mới.
       * `[IsSup]`: Đánh giá câu trả lời tạo ra có được tài liệu chứng minh hay không.
       * `[IsUse]`: Đo lường mức độ hữu ích thực tế với người dùng.
     * *Mô hình tham chiếu:* `selfrag/selfrag_llama2_7b` hoặc sử dụng prompting có cấu trúc trên `Llama-3.1-70B`.
  2. **FLARE Method (Forward-Looking Active Retrieval):**
     * Hệ thống để LLM tạo sinh văn bản câu trả lời. Trong quá trình sinh từng từ, hệ thống theo dõi xác suất xác thực (token probability).
     * Nếu xác suất của một cụm từ dự đoán bị tụt xuống dưới ngưỡng quy định ($\text{Probability} < \delta$, ví dụ $\delta = 0.65$), hệ thống nhận biết LLM đang có nguy cơ bịa đặt (*hallucinating*). Ngay lập tức, luồng sinh bị tạm dừng, cụm từ mơ hồ đó được biến thành truy vấn mới nạp vào Search Module để lấy tài liệu cứu nguy rồi mới tiếp tục sinh.

### 4.2. Iterative Pattern (Lặp lại Luân phiên theo Phản hồi)

* **Nguyên lý kích hoạt:** Dựa trên **Độ đầy đủ của câu trả lời (Information Completeness)** qua từng chu kỳ.
* **Cơ chế hoạt động (Mô hình ITER-RETGEN / ReAct Loop):**
  * Thích hợp cho tác vụ tổng hợp báo cáo chuyên sâu, nghiên cứu thị trường, hoặc viết luận án.
  * **Chu kỳ $t$:**
    1. Mô-đun Search tìm kiếm tài liệu đợt 1.
    2. Mô-đun Generation viết bản thảo đoạn 1 ($Draft_1$).
    3. Bộ thẩm định kiểm tra: $Draft_1$ còn thiếu thông tin gì?
    4. Trích xuất chính lỗ hổng tri thức trong $Draft_1$ thành truy vấn mới $Q_2$.
    5. Kích hoạt Search Module đợt 2, bổ sung tài liệu để hoàn thiện đoạn 2 ($Draft_2$).
  * **Điều kiện dừng (Stopping Criteria):**
    * Số vòng lặp đạt ngưỡng an toàn ($Iteration = Max\_Iterations$, thường đặt từ 3 đến 5).
    * Bộ đánh giá (Evaluator) xác nhận toàn bộ các tiêu chí trong prompt gốc đã được thỏa mãn đầy đủ.

### 4.3. Recursive Pattern (Đệ quy theo Chiều sâu Cây Suy luận)

* **Nguyên lý kích hoạt:** Dựa trên **Mức độ phụ thuộc đa bước (Multi-hop Reasoning / Question Ambiguity)**.
* **Cơ chế hoạt động (IRCoT - Interleaving Retrieval with Chain-of-Thought):**
  * Dành riêng cho các bài toán phức tạp đòi hỏi thông tin bắc cầu (Ví dụ: *"Giám đốc điều hành của công ty sản xuất kính cho kính thiên văn James Webb tốt nghiệp trường đại học nào?"* $\rightarrow$ Không thể tìm trực tiếp bằng một câu query).
  * **Tiến trình đệ quy:**
    1. Rewrite Module phân tích và tách câu hỏi thành cây phụ thuộc:
       $$Q \longrightarrow Q_1 \longrightarrow Q_2 \longrightarrow Q_3$$
    2. Tìm kiếm cho $Q_1$: *"Ai sản xuất kính cho kính thiên văn James Webb?"* $\rightarrow$ Kết quả: *Công ty Coherent Corp*.
    3. Cập nhật ngữ cảnh, sinh đệ quy $Q_2$: *"Ai là Giám đốc điều hành của Coherent Corp?"* $\rightarrow$ Kết quả: *Vincent D. Mattera, Jr.*
    4. Cập nhật ngữ cảnh, sinh đệ quy $Q_3$: *"Vincent D. Mattera, Jr. tốt nghiệp trường đại học nào?"* $\rightarrow$ Kết quả: *Đại học Brown*.
  * **Điều kiện dừng:**
    * Cây suy luận chạm tới lá cuối cùng ($Depth = Max\_Depth$).
    * Trích xuất được thực thể đích trả lời trọn vẹn câu hỏi ban đầu.

---

## 5. Bảng Tổng hợp Tech Stack Khuyến nghị cho Doanh nghiệp

| Hạng mục / Khối | Công nghệ Mã nguồn mở (Self-hosted) | Giải pháp SaaS / Cloud Enterprise |
| :--- | :--- | :--- |
| **Framework Điều phối** | **LangGraph**, **LlamaIndex Workflows** | **AWS Bedrock Agents**, **Azure AI Foundry** |
| **Vector Database** | **Qdrant**, **Milvus**, **pgvector** | **Pinecone Serverless**, **Weaviate Cloud** |
| **Knowledge Graph** | **Neo4j Community**, **Memgraph** | **Neo4j AuraDB Enterprise** |
| **Embedding Model** | `BAAI/bge-m3`, `nomic-embed-text-v1.5` | `text-embedding-3-large` (OpenAI), Cohere Embed |
| **Reranker Model** | `BAAI/bge-reranker-v2-m3` | **Cohere Rerank v3.5** |
| **Mô-đun Định tuyến** | **Semantic Router** (Aurelio) | GPT-4o-mini Tool Calling |
| **Mô-đun Bộ nhớ** | **Mem0**, **Letta (MemGPT)** | **Zep Cloud** |
| **LLM Reasoning & Gen** | `Qwen2.5-72B-Instruct`, `Llama-3.1-70B` | `Claude 3.5 Sonnet`, `GPT-4o` |

---

## 6. Lộ trình Triển khai Thực tế (Production Roadmap)

1. **Bước 1 (PoC nhanh):** Bắt đầu với **Advanced RAG** cơ bản (Small2Big Chunking + Hybrid Search BM25/Vector + BGE-Reranker).
2. **Bước 2 (Kiểm soát chi phí & độ trễ):** Bổ sung **Semantic Router** để chuyển hướng câu hỏi xã giao và câu hỏi đơn giản, tránh gọi RAG tốn kém.
3. **Bước 3 (Xử lý bài toán phức tạp):** Đóng gói thành các mô-đun độc lập trên **LangGraph**. Tích hợp cơ chế **Adaptive (Self-RAG)** hoặc **Recursive (Multi-hop)** cho các nghiệp vụ kiểm toán, pháp lý, kỹ thuật cao cấp.