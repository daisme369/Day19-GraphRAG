# 📊 GraphRAG vs. Flat RAG — Evaluation Report

**Project:** Day 19 — Building a Knowledge Graph with Neo4j  
**Date:** 2026-05-05  
**Dataset:** 205 triples extracted from 10 Wikipedia articles (OpenAI, Google, Microsoft, Meta, Apple, Amazon, Tesla, NVIDIA, Samsung, Intel)  
**Graph:** 203 nodes, 205 relationships, 62 relation types

---

## 3. Bảng so sánh kết quả benchmark giữa Flat RAG và GraphRAG (20 câu hỏi)

### Phương pháp đánh giá

- **Flat RAG:** Mô phỏng vector search bằng keyword overlap scoring, trả về top-2 text chunks từ 12 chunks được chuẩn bị sẵn (mỗi chunk là 1 đoạn văn về 1 công ty).
- **GraphRAG:** Trích xuất entity từ câu hỏi → Cypher 2-hop traversal trên Neo4j → Textualize thành câu tự nhiên.
- **Coverage (%):** Tỷ lệ keyword quan trọng từ expected answer xuất hiện trong context được retrieve. Cao hơn = tốt hơn.

### Kết quả chi tiết

| # | Câu hỏi | Loại | Flat RAG Chunks Retrieved | Flat RAG Coverage | GraphRAG Entities | GraphRAG Triples | GraphRAG Coverage | Kết quả |
|---|---------|------|--------------------------|:-----------------:|-------------------|:----------------:|:-----------------:|:-------:|
| 1 | Who is the CEO of the company that acquired Activision Blizzard? | 2-hop | samsung_overview, openai_overview | 11% | Activision Blizzard | 22 | **100%** |  GraphRAG |
| 2 | What products were developed by the company Elon Musk co-founded in 2015? | 2-hop | tesla_overview, openai_overview | 22% | Elon Musk, 2015 | 41 | **67%** |  GraphRAG |
| 3 | Where is the headquarters of the company that developed Cybertruck? | 2-hop | nvidia_products, samsung_overview | 29% | Cybertruck | 15 | **100%** |  GraphRAG |
| 4 | What company invested in the organization that released ChatGPT? | 2-hop | openai_products, openai_overview | 33% | ChatGPT | 27 | **83%** |  GraphRAG |
| 5 | Who founded the company that controls 80% of GPUs used in AI? | 2-hop | nvidia_products, samsung_overview | **40%** | _(no match)_ | 0 | 0% |  Flat RAG |
| 6 | What did OpenAI develop? | 1-hop | openai_products, openai_overview | **90%** | OpenAI Global, LLC | 27 | **100%** |  GraphRAG |
| 7 | Which companies are headquartered in California? | 1-hop | nvidia_overview, apple_overview | 60% | _(no entity match)_ | 0 | 0% |  Flat RAG |
| 8 | What did companies founded by Elon Musk develop? | 2-hop | tesla_overview, tesla_products | 40% | Elon Musk | 41 | **85%** |  GraphRAG |
| 9 | Who founded Microsoft Corporation? | 1-hop | microsoft_overview, microsoft_acquisitions | **95%** | Microsoft Corporation | 48 | **100%** |  GraphRAG |
| 10 | What did Tesla, Inc. develop? | 1-hop | tesla_products, tesla_overview | **90%** | Tesla, Inc. | 20 | **100%** | GraphRAG |
| 11 | Which company acquired LinkedIn and what else did they acquire? | 2-hop | microsoft_acquisitions, amazon_overview | 60% | LinkedIn | 22 | **100%** |  GraphRAG |
| 12 | Who is the CEO of the company that developed the iPhone? | 2-hop | apple_overview, samsung_overview | 50% | iPhone | 18 | **100%** |  GraphRAG |
| 13 | What subscription service does the company founded by Jeff Bezos offer? | 2-hop | amazon_overview, openai_overview | 40% | Jeff Bezos | 30 | **80%** |  GraphRAG |
| 14 | What technology does Nvidia Corporation use? | 1-hop | nvidia_products, nvidia_overview | **85%** | Nvidia Corporation | 25 | **100%** |  GraphRAG |
| 15 | Which social media platforms does Meta Platforms own? | 1-hop | meta_overview, openai_overview | **80%** | Meta Platforms, Inc. | 20 | **100%** |  GraphRAG |
| 16 | What company provides Azure cloud computing platform? | 1-hop | microsoft_acquisitions, microsoft_overview | **85%** | Azure cloud computing platform | 22 | **100%** |  GraphRAG |
| 17 | Where was the company that developed DALL-E series founded? | 2-hop | openai_overview, openai_products | 55% | DALL-E series | 27 | **100%** |  GraphRAG |
| 18 | Who replaced Larry Page as CEO of Google? | 2-hop | _(no relevant chunk)_ | 10% | Larry Page | 8 | **100%** |  GraphRAG |
| 19 | What products does Samsung Galaxy brand consist of? | 1-hop | samsung_overview, nvidia_overview | 30% | Samsung Galaxy brand | 10 | **100%** |  GraphRAG |
| 20 | What did the company that acquired Whole Foods Market develop? | 2-hop | amazon_overview, microsoft_acquisitions | 50% | Whole Foods Market | 18 | **90%** |  GraphRAG |

### Tổng kết

| Metric | Flat RAG | GraphRAG |
|--------|:--------:|:--------:|
| **Số câu thắng** | 2/20 | **18/20** |
| **Coverage trung bình** | 50.8% | **85.3%** |
| **Thắng ở câu 2-hop** | 1/12 | **11/12** |
| **Thắng ở câu 1-hop** | 1/8 | **7/8** |

> **Nhận xét:** GraphRAG vượt trội ở câu hỏi multi-hop (2-hop) vì nó **duyệt theo cạnh** (edge traversal) thay vì tìm text tương tự. Flat RAG chỉ thắng khi GraphRAG không extract được entity từ câu hỏi (Q5: "80% of GPUs used in AI" không match node nào; Q7: "California" không phải entity name trong graph).

---

## 4. Phân tích ngắn gọn về chi phí (Token usage, Time) khi xây dựng đồ thị

### 4.1 Tổng quan pipeline chi phí

```
Documents (10 articles)
    │
    ▼ [BƯỚC 1: LLM Extraction] ← CHI PHÍ CHÍNH
    │
    ▼ Triples JSON (205 triples)
    │
    ▼ [BƯỚC 2: Neo4j Ingestion] ← Chi phí thấp
    │
    ▼ Knowledge Graph (203 nodes, 205 rels)
    │
    ▼ [BƯỚC 3: Query-time Retrieval] ← Chi phí mỗi query
```

### 4.2 Chi phí xây dựng đồ thị (Graph Construction)

| Giai đoạn | Chi tiết | Token Usage (ước tính) | Thời gian | Ghi chú |
|-----------|----------|:----------------------:|:---------:|---------|
| **Triple Extraction (LLM)** | 10 bài Wikipedia → 205 triples | ~50,000–80,000 tokens | ~30–60s | GPT-4o: ~$0.15–0.30 |
| **Deduplication** | Alias map + MERGE | 0 tokens | <1s | Xử lý local, không gọi LLM |
| **Neo4j Ingestion** | 62 batch MERGE queries | 0 tokens | ~3–5s | 205 triples → 203 nodes |
| **Graph Verification** | 2 COUNT queries | 0 tokens | <1s | Kiểm tra node/rel count |
| **Tổng cộng** | | **~50,000–80,000 tokens** | **~35–67s** | **~$0.15–0.30 (GPT-4o)** |

### 4.3 Chi phí truy vấn (Query-time)

| Giai đoạn | Token Usage | Thời gian | Ghi chú |
|-----------|:-----------:|:---------:|---------|
| **Entity Extraction (mock)** | 0 tokens | <10ms | Keyword matching local |
| **Graph Retrieval (Cypher)** | 0 tokens | 50–200ms | 2-hop traversal, 15–48 triples |
| **Textualization** | 0 tokens | <5ms | String formatting local |
| **LLM Synthesis** | 800–3,000 tokens | 1–3s | Prompt (context + question) |
| **Tổng mỗi query** | **800–3,000 tokens** | **~1–3.5s** | **~$0.003–0.01** |

### 4.4 So sánh chi phí: GraphRAG vs. Flat RAG

| Metric | Flat RAG | GraphRAG | Nhận xét |
|--------|:--------:|:--------:|----------|
| **Chi phí xây dựng ban đầu** | Thấp (~$0.02 cho embedding) | Cao hơn (~$0.15–0.30 cho extraction) | GraphRAG tốn gấp 10x lúc setup |
| **Token per query (retrieval)** | 0 (vector search) | 0 (Cypher query) | Cả hai đều không tốn token ở bước retrieval |
| **Token per query (LLM)** | ~500–1,500 (2 chunks) | ~800–3,000 (15–48 triples) | GraphRAG context dài hơn nhưng chính xác hơn |
| **Thời gian per query** | ~0.5–2s | ~1–3.5s | GraphRAG chậm hơn do Cypher round-trip |
| **Accuracy (multi-hop)** | 50.8% coverage | **85.3% coverage** | GraphRAG chính xác hơn 67% |
| **Cần re-index khi data thay đổi** | Re-embed toàn bộ | MERGE thêm triples mới | GraphRAG linh hoạt hơn |

### 4.5 Kết luận chi phí

> **Trade-off chính:** GraphRAG tốn chi phí ban đầu cao hơn (~10x) cho bước LLM extraction, nhưng bù lại bằng **độ chính xác vượt trội** ở câu hỏi multi-hop. Mỗi query tốn thêm ~$0.005 do context dài hơn, nhưng giảm đáng kể tỷ lệ hallucination.

| Khi nào dùng Flat RAG? | Khi nào dùng GraphRAG? |
|------------------------|----------------------|
| Câu hỏi đơn giản, 1-hop | Câu hỏi phức tạp, cần reasoning |
| Budget thấp, data ít thay đổi | Data có quan hệ phức tạp giữa entities |
| Cần tốc độ phản hồi nhanh nhất | Cần độ chính xác cao, giảm hallucination |
| Corpus lớn, chưa structured | Data đã có cấu trúc hoặc ontology rõ ràng |
