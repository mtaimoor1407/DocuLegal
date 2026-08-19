# ⚖️ DocuLegal — Legal Document Intelligence System

A RAG-powered legal document assistant that enables plain-English 
querying of complex legal documents with clause-level precision.

## 🎯 What It Does

Upload any legal PDF (employment contract, rental agreement, 
terms of service, investor agreement) and ask questions in plain 
English. DocuLegal finds the exact clause, explains it clearly, 
and tells you how confident it is.

**Example:**
> *"Can they fire me without giving a reason?"*

> **DocuLegal:** Yes, according to Clause 14.2 (Termination), the 
> employer can terminate your employment at will with 30 days written 
> notice, without needing to provide a reason...
> 
> 📄 Source: contract.pdf — Page 8 — Section 14.2
> 🟢 Confidence: High

---

## ✨ Features

- 📄 Upload one or multiple legal PDFs
- 💬 Ask questions in plain English
- 📌 Answers cite the exact clause and page number
- 🟢 Confidence scoring (High / Medium / Low)
- ⚖️ Flags when professional legal advice is recommended
- ⚠️ Detects ambiguous or unclear clauses
- 🔀 Compare clauses across multiple documents
- 🧠 Conversation memory — ask follow-up questions naturally

---

## 🏗️ Technical Architecture

PDF Upload
│
▼
PyPDFLoader → Documents
│
▼
Parent-Child Chunking
(Parent: 1500 chars | Child: 300 chars)
│
▼
HuggingFace Embeddings
(sentence-transformers/all-MiniLM-L6-v2)
│
▼
Chroma Vector Store (child chunks)

InMemoryStore (parent chunks)
│
▼
Hybrid Retrieval
BM25 (keyword) + Vector (semantic) + RRF fusion
│
▼
Cross-Encoder Reranking
(cross-encoder/ms-marco-MiniLM-L-6-v2)
│
▼
Parent Document Retrieval
(retrieve child, return parent for full context)
│
▼
Groq LLM (openai/gpt-oss-120b)
Structured JSON Output
│
▼
Streamlit UI


## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| LLM | Groq API (openai/gpt-oss-120b) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | Chroma |
| Retrieval | Hybrid BM25 + Vector + Reranking |
| Chunking | Parent-Child Strategy |
| UI | Streamlit |

## 🚀 Advanced RAG Techniques Used

- **Parent-Child Chunking** — Small chunks indexed for precise 
  retrieval, large parent chunks returned for rich LLM context
- **Hybrid Search** — BM25 keyword search + vector similarity 
  combined with Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — Re-scores retrieved chunks for 
  maximum relevance precision
- **History-Aware Retrieval** — Follow-up questions are rephrased 
  using conversation history before retrieval
- **Structured Output** — Pydantic schemas enforce consistent 
  JSON responses with confidence scoring

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/doculegal.git
cd doculegal
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here

Get a free Groq API key at: https://console.groq.com

### 5. Run the application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

doculegal/
├── app.py ← Streamlit UI
├── requirements.txt ← Dependencies
├── .env ← API keys (not committed)
├── .gitignore
├── README.md
├── core/
│ ├── loader.py ← PDF loading
│ ├── chunker.py ← Parent-child chunking
│ ├── vectorstore.py ← Chroma + InMemoryStore
│ ├── retriever.py ← Hybrid + reranking retriever
│ ├── chain.py ← RAG chain + structured output
│ └── prompts.py ← Prompt templates
├── models/
│ └── schemas.py ← Pydantic output schemas
└── utils/
└── helpers.py ← Display utilities


---

## 💡 Example Questions to Try

**Employment Contracts:**
- *"What is the notice period for termination?"*
- *"Can the employer change my salary without consent?"*
- *"Who owns intellectual property I create during employment?"*

**Rental Agreements:**
- *"Who is responsible for maintenance and repairs?"*
- *"What is the penalty for breaking the lease early?"*
- *"Can the landlord enter without giving notice?"*

**Service Agreements:**
- *"What are the payment terms if a deadline is missed?"*
- *"How can either party terminate this contract?"*
- *"What are my confidentiality obligations?"*

---

## ⚠️ Disclaimer

DocuLegal is for **informational purposes only** and is not a 
substitute for professional legal advice. Always consult a qualified 
lawyer for legal decisions.

---

## 👨‍💻 Author

**Muhammad Taimoor**
BS Computer Science — COMSATS University Islamabad, Lahore Campus