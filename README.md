# DocuLegal — Legal Document Intelligence

A **retrieval-augmented generation (RAG)** application for asking plain-English questions about legal PDFs. It retrieves relevant passages and generates a structured answer with source references.

## Features

- Upload one or multiple PDFs.
- Ask questions about clauses and compare information across documents.
- Retrieve short passages while returning larger surrounding context.
- Combine keyword and semantic retrieval with reranking.
- Display page/section references, follow-up answers, and confidence labels.

## Retrieval pipeline

1. Load PDFs with `PyPDFLoader`.
2. Create parent chunks (1,500 characters) and child chunks (300 characters).
3. Embed child chunks using `sentence-transformers/all-MiniLM-L6-v2` and store them in Chroma.
4. Combine BM25 and vector retrieval using reciprocal rank fusion.
5. Rerank candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
6. Return parent context and generate a structured response using the Groq API.

| Component | Technology |
| --- | --- |
| Application | Python, Streamlit |
| Framework | LangChain |
| Storage | Chroma and an in-memory parent-document store |
| Generation | Groq API; documented model: `openai/gpt-oss-120b` |
| Response structure | Pydantic schemas |

Provider model availability and usage quotas depend on the configured account.

## Run locally

```bash
git clone https://github.com/mtaimoor1407/DocuLegal.git
cd DocuLegal
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.

```bash
pip install -r requirements.txt
```

Create `.env` in the repository root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Start the application:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. The first run may download the embedding and reranking models.

## Example workflow

Upload a document, index it, and ask a question such as:

- What notice period does this agreement specify?
- Which clause describes confidentiality obligations?
- What are the payment deadlines and stated late-payment consequences?
- How do the termination clauses differ between these two documents?

Check the cited passage in the original PDF. These are example prompts, rather than a claim that every uploaded document contains the requested information.

## Repository guide

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `core/` | PDF loading, chunking, retrieval, generation, and prompts |
| `models/schemas.py` | Structured response definitions |
| `utils/` | Supporting display/utilities |
| `requirements.txt` | Python dependencies |
| `.streamlit/` | Streamlit configuration |

## Evaluation and limitations

The confidence label is an application output, rather than a validated probability that an answer is correct. Source references can also be incomplete or incorrect and should be checked.

PDF extraction quality, missing context, retrieval errors, and ambiguous clauses can affect responses. Document comparison depends on retrieving the relevant evidence from each document.

A reproducible evaluation should include answerable and unanswerable questions, expected supporting passages, citation checks, retrieval Recall@k, answer correctness, and latency. Compare keyword-only, vector-only, hybrid, and reranked retrieval before attributing improvements to a specific technique. Quantitative results should be reported after those experiments are run.

DocuLegal is an informational prototype and does not replace professional legal advice.

**Author:** Muhammad Taimoor — BS Computer Science, COMSATS University Islamabad, Lahore Campus.
