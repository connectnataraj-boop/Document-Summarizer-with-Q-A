# Document Summarizer & Q&A — LangGraph Agentic Pipeline

> **LangGraph · LLaMA 3.3 70B · Groq · OpenAI Embeddings · Pinecone · Streamlit · RAG · Agentic AI**

An agentic document Q&A system built with LangGraph that automatically loads a PDF, chunks and embeds it into Pinecone, generates a summary using LLaMA 3.3 70B via Groq, then enters a stateful Q&A loop with quality feedback — all orchestrated as a LangGraph state machine.

---

## 📌 Table of Contents

- [Why I Built This](#-why-i-built-this)
- [Architecture](#-architecture)
- [LangGraph State Machine — Full Flow](#-langgraph-state-machine--full-flow)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Node-by-Node Breakdown](#-node-by-node-breakdown)
- [Conditional Edges Explained](#-conditional-edges-explained)
- [State Design](#-state-design)
- [Sample Q&A](#-sample-qa)
- [Key Learnings](#-key-learnings)
- [Setup Instructions](#-setup-instructions)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🤔 Why I Built This

My previous RAG projects (HR Policy Chatbot with and without LangChain) were **stateless pipelines** — each query was independent, with no memory of previous questions and no ability to loop, retry, or make decisions between steps.

This project adds **agentic behavior on top of RAG** using LangGraph:

| Previous RAG Projects | This Project |
|---|---|
| Stateless — one query at a time | Stateful — remembers full conversation |
| Linear pipeline — no decisions | Graph — conditional routing between nodes |
| No retry logic | Quality check node with 2-retry loop |
| No session memory | `chat_history` grows across turns |
| Terminal only | Streamlit UI with feedback buttons |

LangGraph treats the entire pipeline as a **state machine** — each step (node) reads from shared state, updates it, and the graph decides the next step based on conditions. This is how production agentic AI systems are actually built.

---

## 🏗️ Architecture

### Full Pipeline Overview

```
PDF File
   │
   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  load_pdf   │────▶│  chunk_text  │────▶│ embed_store │────▶│  summarize   │
│             │     │              │     │             │     │              │
│ PdfReader   │     │ Recursive    │     │ OpenAI      │     │ LLaMA 3.3    │
│ extracts    │     │ CharText     │     │ embeddings  │     │ 70B via Groq │
│ raw text    │     │ Splitter     │     │ → Pinecone  │     │ generates    │
│             │     │ 500/150      │     │ upsert      │     │ summary      │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                     │
                                                                     ▼
                                                           ┌──────────────────┐
                                                           │  get_user_input  │◀──────────────┐
                                                           │                  │               │
                                                           │ Streamlit:       │               │
                                                           │ st.chat_input()  │               │
                                                           └────────┬─────────┘               │
                                                                    │                         │
                                                         ┌──────────▼──────────┐              │
                                                         │  CONDITIONAL EDGE   │              │
                                                         │  router()           │              │
                                                         └──────────┬──────────┘              │
                                                          /          \                        │
                                                    "query"        "exit"                     │
                                                       /                \                     │
                                                      ▼                  ▼                    │
                                           ┌──────────────────┐      ┌──────┐                 │
                                           │  query_process   │      │ END  │                 │
                                           │                  │      └──────┘                 │
                                           │ Pinecone search  │                               │
                                           │ top-k=4 chunks   │                               │
                                           │ LLaMA answers    │                               │
                                           └────────┬─────────┘                               │
                                                    │                                         │
                                                    ▼                                         │
                                           ┌──────────────────┐                               │
                                           │  quality_check   │                               │
                                           │                  │                               │
                                           │ Streamlit:       │                               │
                                           │ 👍 Yes / 👎 No  │                               │
                                           └────────┬─────────┘                               │
                                                    │                                         │
                                         ┌──────────▼──────────┐                              │
                                         │  CONDITIONAL EDGE   │                              │
                                         │  quality_router()   │                              │
                                         └──────────┬──────────┘                              │
                                          /          \                                        │
                                       "good"      "retry"                                   │
                                         /               \                                  │
                                        ▼                 ▼                                 │
                               ┌────────────────┐  ┌──────────────────┐                     │
                               │ get_user_input │  │  quality_check   │──(max 2)────────────┘
                               └────────────────┘  │  (retry loop)    │
                                       │            └──────────────────┘
                                       └────────────────────────────────────────────────────┘
                                            (loops back for next question)
```

---

## 🔄 LangGraph State Machine — Full Flow

### Nodes (7 total)

| Node | Type | What It Does |
|---|---|---|
| `load_pdf` | Processing | Extracts raw text from PDF using PdfReader |
| `chunk_text` | Processing | Splits text into 500-char chunks with 150-char overlap |
| `embed_store` | Processing | Embeds chunks via OpenAI → upserts to Pinecone |
| `summarize` | LLM | LLaMA 3.3 70B summarizes document |
| `get_user_input` | Human-in-loop | Waits for user question (Streamlit chat input) |
| `query_process` | RAG + LLM | Retrieves top-4 chunks → LLaMA answers from context |
| `quality_check` | Human-in-loop | User rates answer → retry or move on |

### Edges (5 total)

```
START          → load_pdf         (unconditional)
load_pdf       → chunk_text       (unconditional)
chunk_text     → embed_store      (unconditional)
embed_store    → summarize        (unconditional)
summarize      → get_user_input   (unconditional)
query_process  → quality_check    (unconditional)
```

### Conditional Edges (2 total)

```
get_user_input  →  router()         → "query_process" or END
quality_check   →  quality_router() → "get_user_input" or "quality_check"
```

---

## 📂 Project Structure

```
Document-Summarizer-QA/
│
├── document_summarizer.py   # Core LangGraph pipeline — all nodes, edges, graph
├── app.py                # Streamlit UI — replaces input() and print() only
│
├── .env                     # (gitignored) — API keys
├── .gitignore
├── requirements.txt
└── README.md
```

### Design Decision — Two Files

`document_summarizer.py` contains the entire LangGraph logic and can run standalone via terminal. `app_v3.py` is a **thin Streamlit wrapper** — it reuses every function from `document_summarizer.py` exactly as written, only replacing:

- `input()` → `st.chat_input()` / `st.file_uploader()`
- `print()` → `st.write()` / `st.info()`
- `input("yes/no")` → `👍 Yes / 👎 No` buttons

This separation means the core logic is testable independently of the UI.

---

## 🛠️ Tech Stack

| Tool | Role | Why Chosen |
|---|---|---|
| **LangGraph** | Agentic state machine orchestration | Stateful loops, conditional routing, human-in-loop |
| **LLaMA qwen/qwen3.6-27b via Groq** | Summarization + Q&A LLM | Free tier, sub-second inference, powerful open model |
| **OpenAI text-embedding-3-small** | Text vectorization (1536-dim) | Best price/performance for embeddings |
| **Pinecone** | Vector database | Fast cosine similarity search at scale |
| **LangChain** | Text splitting + Pinecone integration | RecursiveCharacterTextSplitter + PineconeVectorStore |
| **PdfReader (pypdf)** | PDF text extraction | Lightweight, no external dependencies |
| **Streamlit** | Web UI | Rapid deployment, session state, chat interface |

---

## 🔍 Node-by-Node Breakdown

### `load_pdf`
```python
def load_pdf(state: NodeData) -> dict:
    reader = PdfReader(state["pdf_path"])
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return {"raw_text": "\n".join(pages).strip()}
```
Reads `pdf_path` from state, extracts text from all pages, filters empty pages, joins into single string. Returns `raw_text` to state.

---

### `chunk_text`
```python
def chunk_text(state: NodeData) -> dict:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    return {"chunks": splitter.split_text(state["raw_text"])}
```
`RecursiveCharacterTextSplitter` splits on paragraphs → sentences → words in order. `chunk_size=500` keeps chunks under token limits. `chunk_overlap=150` ensures context at boundaries is never lost — a topic spanning two chunks will appear in at least one chunk fully.

---

### `embed_store`
```python
def embed_store(state: NodeData) -> dict:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore.from_texts(
        texts=state["chunks"], embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    return {"vectorstore": vectorstore}
```
Converts every chunk into a 1536-dimensional float vector using OpenAI's embedding model. `from_texts()` handles both embedding and Pinecone upsert in a single call. The vectorstore object is stored in state so all subsequent nodes can access it.

---

### `summarize`
```python
def summarize(state: NodeData) -> dict:
    llm = get_llm()
    messages = [
        {"role": "system", "content": "Summarize the following document clearly and concisely."},
        {"role": "user", "content": "\n\n".join(state["chunks"][:20])}
    ]
    response = llm.invoke(messages)
    return {"summary": response.content, "retry_count": 0, "chat_history": []}
```
Uses first 20 chunks to stay within LLaMA's context window. Also initialises `retry_count` and `chat_history` here — these are used throughout the rest of the graph.

---

### `query_process`
```python
def query_process(state: NodeData) -> dict:
    results = state["vectorstore"].similarity_search(state["user_input"], k=4)
    context = "\n\n".join([doc.page_content for doc in results])
    # LLaMA answers from context only
    # appends to chat_history
```
Core RAG node. Embeds the query, finds top-4 semantically similar chunks via cosine similarity, injects them into the LLM prompt. The system message strictly constrains the model to answer only from provided context — preventing hallucination.

---

### `quality_check`
```python
def quality_check(state: NodeData) -> dict:
    # if feedback == "yes" → answer_quality = "good"
    # if feedback == "no" and retry_count < 2 → retry with improved prompt
    # if retry_count >= 2 → force "good" and move on
```
Human-in-the-loop node. When the user rates an answer as poor, it crafts a more detailed prompt referencing the previous unsatisfactory answer. Hard cap at 2 retries prevents infinite loops. In Streamlit, `input()` is replaced by 👍/👎 buttons.

---

## 🔀 Conditional Edges Explained

### `router()` — after `get_user_input`
```python
def router(state: NodeData) -> Literal["query_process", "end"]:
    if state["user_input"].strip().lower() == "exit":
        return "end"    # → END node
    return "query_process"  # → answer the question
```
Checks user intent. "exit" terminates the graph cleanly. Any other input proceeds to retrieval and answering.

---

### `quality_router()` — after `quality_check`
```python
def quality_router(state: NodeData) -> Literal["get_user_input", "quality_check"]:
    if state["answer_quality"] == "good":
        return "get_user_input"   # loop back — next question
    return "quality_check"        # retry — same question, better answer
```
Creates the retry loop. When `answer_quality == "good"` (either user said yes, or max retries reached), the graph loops back to `get_user_input` for the next question. When `answer_quality == "retry"`, it cycles back through `quality_check` for a better attempt.

---

## 🗂️ State Design

```python
class NodeData(TypedDict):
    # Document pipeline
    pdf_path: str           # input — path to PDF
    raw_text: str           # set by load_pdf
    chunks: List[str]       # set by chunk_text
    summary: str            # set by summarize
    vectorstore: object     # set by embed_store — used by query_process

    # Q&A loop
    user_input: str         # set by get_user_input
    intent: str             # "question" or "exit"
    retrieved_chunks: List[str]  # set by query_process — used by quality_check retry
    answer: str             # set by query_process, updated by quality_check
    answer_quality: str     # "good" or "retry" — drives quality_router
    retry_count: int        # incremented in quality_check, reset in summarize

    # Memory
    chat_history: List[dict]  # grows with every Q&A turn
```

**Key design decisions:**
- `vectorstore` stored in state so `query_process` doesn't need to rebuild it every call
- `retrieved_chunks` stored in state so `quality_check` can use the same context for retry without re-querying Pinecone
- `chat_history` accumulates across the entire session — enables future multi-turn context

---

## 💬 Sample Q&A

Example with an HR Policy PDF loaded:

---

**📋 Summary:**
> The document outlines the company's HR policies covering leave entitlements, work-from-home guidelines, travel reimbursements, resignation procedures, and code of conduct. Employees are entitled to 12 days casual leave and 26 weeks maternity leave annually...

---

**Q: What is the notice period for resignation?**
> A: The notice period is 30 days for employees below manager level and 60 days for manager and above. The company may waive the notice period at its discretion.

**Was this answer helpful? 👍 Yes**

---

**Q: What travel expenses are reimbursable?**
> A: Business travel expenses including economy airfare, hotel accommodation up to ₹4,000/night, and a daily meal allowance of ₹500 are reimbursable. All receipts must be submitted within 7 days of travel completion.

**Was this answer helpful? 👎 No — Retry**

> **Improved Answer:** The travel reimbursement policy covers the following in detail: (1) Airfare — economy class only, business class requires VP approval; (2) Accommodation — up to ₹4,000 per night, any excess requires manager pre-approval; (3) Daily allowance — ₹500 for meals when travelling outside the city; (4) Local transport — actual costs by auto/cab are reimbursable with receipts; (5) Submission deadline — all claims must be filed within 7 working days with original receipts attached...

---

> **Note:** Answers depend on the PDF you upload. The system strictly answers from document context only.

---

## 🧠 Key Learnings

### 1. LangGraph vs plain LangChain — when agentic adds value
LangChain is great for linear pipelines. LangGraph is needed when you have **loops**, **conditional branching**, or **human-in-the-loop** steps. This project needs all three — making it a genuine use case for LangGraph rather than a forced choice.

### 2. State is the backbone of agentic systems
Every node reads from and writes to shared `NodeData` state. This means `quality_check` can access `retrieved_chunks` set by `query_process` without any function arguments — the graph wires it automatically. Understanding state design is what separates agentic engineers from LangChain tutorial followers.

### 3. `vectorstore` in state prevents redundant re-embedding
A naive approach would re-embed and re-query Pinecone from scratch every turn. Storing the vectorstore object in state means it's built once in `embed_store` and reused for every query. At scale this is a significant cost and latency saving.

### 4. Conditional edges are just routing functions
`router()` and `quality_router()` are plain Python functions that return a string. LangGraph matches that string to a node name in the edges map. Once you understand this, conditional routing becomes intuitive — it's just an if/else that returns a destination.

### 5. Human-in-the-loop requires UI replacement strategy
The terminal version uses `input()` for human feedback. Streamlit can't use `input()` — it blocks the event loop. The solution is to store state in `st.session_state`, show buttons, and use `st.rerun()` to re-execute the script with updated state after each interaction. This pattern applies to any LangGraph human-in-the-loop node deployed in a web app.

### 6. Retry loops need hard caps
The `quality_check` retry loop could theoretically run forever. `retry_count >= 2` forces `answer_quality = "good"` even on a bad answer — breaking the loop. Without this, one persistent "no" from the user would create an infinite cycle. Always cap retry loops in agentic systems.

### 7. LLaMA 3.3 70B via Groq is genuinely production-ready
Sub-second inference, free tier, and quality comparable to GPT-4 on document Q&A tasks. For non-OpenAI deployments this is the best option available as of 2026.

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/connectnataraj-boop/Document-Summarizer-QA.git
cd Document-Summarizer-QA
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or with uv (faster):
```bash
uv add langgraph langchain-groq langchain-openai langchain-pinecone \
       langchain-text-splitters pypdf streamlit python-dotenv
```

### 3. Create `.env` File
```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here
```

**Where to get keys:**
- Groq: https://console.groq.com (free)
- OpenAI: https://platform.openai.com/api-keys
- Pinecone: https://app.pinecone.io → Create Index → dimension: 1536, metric: cosine

### 4. Create Pinecone Index
In your Pinecone dashboard:
- Dimension: `1536` (matches text-embedding-3-small)
- Metric: `cosine`
- Cloud: any (free tier available)

### 5. requirements.txt
```
langgraph
langchain-groq
langchain-openai
langchain-pinecone
langchain-text-splitters
langchain-core
pypdf
streamlit
python-dotenv
```

---

## ▶️ How to Run

### Option 1 — Streamlit UI (recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

```
1. Upload your PDF using the file uploader
2. Click "🚀 Process PDF" — wait for all 4 spinners to complete
3. Read the generated summary
4. Type your question in the chat box at the bottom
5. Rate the answer with 👍 Yes or 👎 No
6. Ask more questions or upload a new PDF
```

### Option 2 — Terminal
```bash
python document_summarizer.py
```
```
Enter the path to your PDF file: C:\Users\you\docs\policy.pdf
✅ PDF loaded
✅ Chunked into 47 chunks
✅ Embedded and stored in Pinecone
📄 DOCUMENT SUMMARY: ...
Ask a question (or type 'exit'):
You: What is the leave policy?
Answer: ...
Is the answer good? (yes/no): yes
You: exit
👋 Goodbye!
```

---


## 👤 Author

**S. Nataraj** — GenAI Engineer · Deep Learning & AI
Tirupur, Tamil Nadu, India
📧 connectnataraj@outlook.com
🔗 [GitHub](https://github.com/connectnataraj-boop) · [LinkedIn](https://linkedin.com/in/nataraj-sb-b5a84a3b7)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> *"My first RAG project had no framework. My second used LangChain. This one adds LangGraph — stateful loops, conditional routing, and human feedback. Each project built on the last, and now I understand the full stack from embeddings to agentic orchestration."*
