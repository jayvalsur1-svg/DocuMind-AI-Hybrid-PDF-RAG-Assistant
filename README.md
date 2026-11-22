# DocuMind AI — Hybrid PDF RAG Assistant

A PDF question-answering application built with **FastAPI, LangChain, FAISS, BM25, Sentence Transformers, and Groq**. Upload or select a PDF, ask questions about its contents, and receive an answer generated from retrieved document context with page-level source references.

DocuMind combines dense semantic retrieval with sparse keyword retrieval. It also uses a separate introductory-page retrieval path for broad questions such as summaries and document overviews.

## Key Features

- **PDF text extraction:** Extracts page text using PyMuPDF (`fitz`) and records page numbers and source filenames as metadata.
- **Text chunking:** Splits extracted content into chunks of 800 characters with 150-character overlap.
- **Semantic search:** Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2` and stores them in a FAISS vector index.
- **Keyword search:** Uses a BM25 retriever to find relevant passages through lexical matching.
- **Hybrid retrieval:** Combines FAISS MMR results and BM25 results, removes duplicate page content, and selects up to six context chunks.
- **Context-aware overview retrieval:** For queries containing terms such as “summary”, “overview”, “topics”, or “table of contents”, retrieves available text from the first six pages.
- **LLM-generated answers:** Uses Groq's `qwen/qwen3.8-27b` model with a prompt that instructs it to answer from the supplied context and cite page numbers.
- **Caching:** Keeps document indexes in memory and saves FAISS indexes and chunk data locally for reuse.
- **FastAPI endpoints:** Provides routes for health checks, document listing, PDF upload, and question answering.

## Architecture

```text
PDF document
    |
    v
PyMuPDF page extraction
    |
    v
Page-aware LangChain Documents
    |
    v
RecursiveCharacterTextSplitter
(800 chunk size / 150 overlap)
    |
    +-----------------------------+
    |                             |
    v                             v
Sentence Transformer          BM25 Retriever
Embeddings                    (keyword search)
    |                             |
    v                             |
FAISS Vector Store                |
    |                             |
    v                             |
MMR semantic retrieval            |
    |                             |
    +-------------+---------------+
                  |
                  v
       Combine and deduplicate
       retrieved document chunks
                  |
                  v
        Context + user question
                  |
                  v
        Groq Qwen language model
                  |
                  v
       Answer with page references
```

For overview-style queries, the retrieval function instead uses available text from the first six pages.

## Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| PDF extraction | PyMuPDF (`fitz`) |
| LLM orchestration | LangChain |
| Language model | Groq — `qwen/qwen3.8-27b` |
| Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Dense vector search | FAISS |
| Sparse keyword search | BM25 |
| Text splitting | RecursiveCharacterTextSplitter |
| Data validation | Pydantic |
| Local persistence | FAISS index + Python pickle |

## Project Structure

The backend code expects `frontend.html`, `.env`, PDF files, and the cache directory to be located relative to the Python file.

```text
DocuMind-AI/
├── main.py
├── frontend.html
├── .env
├── *.pdf
└── .faiss_cache/
    └── <document-cache-name>/
        ├── index.faiss
        ├── index.pkl
        └── chunks.pkl
```

`main.py` is a suggested filename for the backend script; adjust the command below if your file has a different name. The frontend file is served by the backend, but its implementation was not included in the supplied backend code.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DocuMind-AI.git
cd DocuMind-AI
```

Replace the repository URL with your actual GitHub repository.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn python-multipart python-dotenv pymupdf langchain langchain-groq langchain-core langchain-huggingface langchain-text-splitters langchain-community sentence-transformers faiss-cpu rank-bm25
```

Depending on the installed LangChain package versions and your Python environment, you may need to adjust package versions. For reproducible deployment, pin and test the exact versions in a `requirements.txt` file.

### 4. Configure the API key

Create a `.env` file in the same directory as the backend script:

```env
GROQ_API_KEY=your_groq_api_key
```

Keep `.env` private. Do not commit API keys to GitHub.

### 5. Add PDF documents

Place PDF files in the backend directory. The upload endpoint can also save a PDF into this directory.

### 6. Run the application

```bash
python main.py
```

The supplied script starts Uvicorn on `127.0.0.1:8000` and attempts to open the frontend in a browser.

## API Reference

Base URL when running locally:

```text
http://127.0.0.1:8000
```

### `GET /`

Serves `frontend.html` from the backend directory. Returns a 404 HTML response if the file is missing.

### `GET /frontend.html`

Alias for the frontend route.

### `GET /health`

Returns application status, the selected default PDF, the filenames currently indexed in memory, and the total number of PDFs found in the backend directory.

Example response:

```json
{
  "status": "ok",
  "default_document": "The Ultimate Python Handbook.pdf",
  "indexed_documents": [],
  "total_documents": 1
}
```

### `GET /documents`

Lists PDF files found in the backend directory, including their filename, size, page count, chunk count, and whether an index is available.

### `POST /upload`

Uploads a PDF using multipart form data with the field name `file`. The backend checks that the filename ends in `.pdf`, saves the file, and indexes it.

Example using cURL:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@document.pdf"
```

A successful response includes the saved filename, page count, and chunk count.

### `POST /ask`

Asks a question about a PDF. The `filename` field is optional. If omitted, the backend selects `The Ultimate Python Handbook.pdf` when present; otherwise, it selects the first PDF returned by the directory listing.

Request:

```json
{
  "question": "Explain the main topics covered in this document.",
  "filename": "document.pdf"
}
```

Example response shape:

```json
{
  "answer": "Answer generated from the retrieved document context...",
  "question": "Explain the main topics covered in this document.",
  "filename": "document.pdf",
  "sources": [
    {
      "page": 1,
      "snippet": "A short excerpt from the retrieved page..."
    }
  ]
}
```

The exact answer and source entries depend on the PDF and the retrieval results.

## Retrieval Strategy

DocuMind uses two retrieval methods for regular questions:

1. **Dense retrieval:** FAISS performs semantic similarity search using sentence-transformer embeddings. Its retriever is configured for MMR with `k=4` and `fetch_k=12`.
2. **Sparse retrieval:** BM25 performs keyword-based retrieval with `k=4`.

The results are concatenated, deduplicated by exact `page_content`, and limited to six context entries before being sent to the LLM. This is a practical hybrid retrieval approach; the current implementation does not perform score normalization, weighted ranking, or Reciprocal Rank Fusion (RRF).

For overview-style questions, the code checks for selected keywords and uses text from the first six pages when available. This is a heuristic based on query wording, not a semantic query classifier.

## Caching and Persistence

- Indexed documents are stored in the in-memory `document_cache` dictionary for the lifetime of the running process.
- FAISS indexes are saved under `.faiss_cache/`.
- Chunked LangChain documents are serialized to `chunks.pkl`.
- On a later load, the application attempts to restore the FAISS index and chunk data, then reconstructs the BM25 and MMR retrievers.

The cache is local to the application directory. It is not a shared database or distributed cache.

## Current Limitations and Security Considerations

The following points are based on the supplied backend implementation and should be addressed before exposing the application to untrusted users or deploying it publicly:

- **Upload filename handling:** The upload endpoint joins the client-supplied filename directly to `BASE_DIR`. A malicious filename may attempt path traversal. Generate a server-side filename, normalize it, and verify the resolved path remains inside the intended upload directory.
- **PDF validation:** Checking the `.pdf` extension does not prove that the uploaded content is a valid or safe PDF. Validate the file, handle malformed documents, and consider malware scanning.
- **Upload size limits:** The current upload handler reads the entire file into memory and does not enforce a maximum file size. Add request and file-size limits.
- **Pickle deserialization:** The cache loader uses `pickle.load()` and FAISS loading with `allow_dangerous_deserialization=True`. Only load cache files created and protected by the trusted application. Do not accept cache files from users or untrusted sources.
- **CORS configuration:** The application allows all origins. Configure an explicit allowlist for production.
- **In-memory state:** Document cache entries are process-local and disappear when the server restarts. There is no multi-user cache coordination.
- **Retrieval ranking:** Dense and sparse results are concatenated rather than fused using a ranking algorithm. The first six unique results are retained.
- **Overview heuristic:** Overview retrieval is triggered by a fixed list of words and uses the first six pages; it may not fit every PDF or every summary request.
- **LLM grounding:** The prompt requests context-only answers and page citations, but the backend does not independently verify that every generated claim or citation is supported.
- **Scanned PDFs:** The code extracts embedded text with PyMuPDF. It does not implement OCR, so image-only or scanned pages may not be searchable.
- **No authentication or rate limiting:** The supplied backend does not include user authentication, authorization, or request throttling.

## Potential Improvements

- Add OCR support for scanned PDFs.
- Implement Reciprocal Rank Fusion (RRF) or another explicit dense/sparse ranking strategy.
- Add reranking to improve the relevance of retrieved chunks.
- Use a persistent metadata database and a safer cache format.
- Add secure upload handling, file-size limits, authentication, and rate limiting.
- Add automated tests for PDF parsing, retrieval, caching, and API routes.
- Add structured logging and request-level error handling.
- Add streaming responses for long answers.
- Add document deletion and cache invalidation endpoints.
- Evaluate retrieval quality with a repeatable question-and-answer test set.

## Project Scope

DocuMind AI is a **retrieval-augmented generation (RAG) application** for querying PDF documents. It combines semantic and keyword retrieval and passes selected document excerpts to an LLM to generate answers. The supplied implementation is a document QA backend; it does not implement an autonomous multi-step agent workflow.

## License

Choose and add a license before publishing this repository. If you have not selected one, do not imply that the project is open source under a particular license.
