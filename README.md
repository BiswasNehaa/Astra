# 🛰️ ASTRA — Self-Correcting RAG Research Assistant

![CI](https://github.com/BiswasNehaa/Astra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

> A Retrieval-Augmented Generation system that searches, retrieves, and answers questions from academic paper abstracts on arXiv — and checks its own answers before returning them.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why This Project Exists](#-why-this-project-exists)
3. [Tech Stack & Why Each Tool Was Chosen](#-tech-stack--why-each-tool-was-chosen)
4. [System Architecture — The Big Picture](#-system-architecture--the-big-picture)
5. [File Structure](#-file-structure)
6. [Complete Pipeline Walkthrough](#-complete-pipeline-walkthrough)
7. [Every Function Explained](#-every-function-explained)
8. [Setup & Installation](#-setup--installation)
9. [Example Input & Output](#-example-input--output)
10. [Design Decisions & Tradeoffs](#-design-decisions--tradeoffs)
11. [Known Limitations](#-known-limitations)
12. [Future Enhancements](#-future-enhancements)

---

## 🧠 Project Overview

ASTRA is a RAG (Retrieval-Augmented Generation) system that lets you ask questions about academic research and get answers grounded in real, retrieved paper abstracts — not the AI's own memory. What makes it different from a basic "chat with your papers" project is a **verification step**: after generating an answer, a second AI call independently checks whether that answer is actually supported by the retrieved text. If it isn't, the system automatically retries with a refined search — up to twice — before honestly returning its best answer.

The core design goal: **never let the system confidently state something its sources don't actually support.**

---

## 💡 Why This Project Exists

Generic AI chatbots answer research questions from their own training data — which means they can hallucinate specific facts, misattribute findings, or state things confidently that are simply wrong. In any serious research context, that's a real problem, not a minor inconvenience.

ASTRA addresses this by:
- Only answering from **actually retrieved** text, not general AI knowledge
- **Independently fact-checking its own output** against those sources before returning it
- Being **honest when it doesn't know** rather than filling gaps with plausible-sounding guesses

---

## 🛠 Tech Stack & Why Each Tool Was Chosen

| Tool | Purpose | Why This and Not Something Else |
|---|---|---|
| **Python** | Core language | Standard for AI/ML work, huge ecosystem |
| **FastAPI** | Web API framework | Async-friendly, automatic docs (`/docs`), minimal boilerplate compared to Flask |
| **LangGraph** | Orchestrates the self-correction loop | Regular LangChain chains only go in a straight line; LangGraph supports **conditional loops** — essential for "retry if unsupported" logic |
| **Groq (`openai/gpt-oss-120b`)** | LLM for generation and verification | Free tier, very fast inference, strong enough for both answering and fact-checking |
| **sentence-transformers (`bge-small-en-v1.5`)** | Embedding model | Runs locally, free, no API cost; BGE is specifically trained for retrieval tasks (as opposed to general-purpose embedding models) |
| **ChromaDB** | Vector database | Simpler than FAISS/Qdrant for a project this size — built-in persistence and metadata filtering with far less setup code |
| **arXiv API** | Data source | Free, official, no scraping — but abstracts only (see Design Decisions) |
| **python-dotenv** | Secrets management | Keeps API keys out of source code and git history |
| **Docker** | Containerization | Makes the app runnable identically anywhere, not just "on my machine" |

---

## 🏗 System Architecture — The Big Picture

```
User Question
      │
      ▼
┌─────────────────────────────┐
│ Stage A — Router            │
│ Pass question to retrieval  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Stage B — Semantic Retrieval│
│ Question → Embedding        │
│          → ChromaDB         │
│          → Top-K chunks     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Stage C — Answer Generation │
│ Question + Retrieved Chunks │
│          → LLM              │
│          → Draft Answer     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Stage D — Verification      │
│ Draft + Retrieved Chunks    │
│          → LLM              │
│          → Supported?       │
└──────────────┬──────────────┘
               │
        ┌──────┴───────┐
        ▼              ▼
    Supported?      Not Supported
        │              │
        ▼              ▼
 Return Answer     Retry C → D
                       │
                  Max 2 Attempts
                       │
                       ▼
                 Return Best Answer
```

**The crucial insight:** Here the LLM does both the answering *and* the checking — but as two **separate, independent calls** with different, narrow instructions. The verifier is never told "you just wrote this, was it good?" — it's given the answer and the sources fresh, and asked a strict, narrow yes/no question. This separation is what keeps the check honest instead of the AI just agreeing with itself.

---

## 📁 File Structure

```
Astra/
├── main.py          # FastAPI app, /ask, /ingest, and /summarize_topic endpoints
├── graph.py          # LangGraph pipeline: generate → verify → retry/cite
├── rag.py            # Original single-pass RAG (superseded, kept for reference)
├── llm.py             # ask_ai() — wraps the Groq API call
├── embeddings.py       # get_embedding() — BGE embeddings
├── vectorstore.py       # add_chunk(), search() — Chroma operations
├── ingestion.py          # fetch_papers(), chunk_text(), ingest_papers(), summarize_paper(), summarize_topic()
├── config.py              # Loads GROQ_API_KEY from .env
├── static/index.html       # Minimal demo UI, served at /ui/
├── data/chroma/            # Auto-generated vector DB (gitignored)
├── requirements.txt
├── Dockerfile
└── LICENSE

```
---

## 🔬 Complete Pipeline Walkthrough

### Stage A — Ingestion (`/ingest`, run before asking questions)

**What it does:** Takes a topic string (e.g., `"quantum computing"`), fetches matching paper abstracts from arXiv, breaks each abstract into overlapping ~100-word chunks, converts each chunk into an embedding, and stores it in Chroma with metadata (title, URL, chunk position).

**Why chunking matters:** A full abstract's embedding blurs together multiple ideas into one vector, making search less precise. Smaller chunks give sharper, more targeted retrieval — and the AI gets more focused context instead of a wall of text.

**Why overlap:** Each new chunk repeats the last ~20 words of the previous one, so a sentence or idea sitting right at a chunk boundary doesn't get cut in half and lost.

---

### Stage B — Semantic Retrieval (inside `/ask`)

**What it does:** Converts the user's question into an embedding using the *same* BGE model used during ingestion, then asks Chroma for the stored chunks whose embeddings are closest in meaning — not closest in exact wording.

**Why this matters:** A question like "how do you move a spacecraft between two orbits" correctly retrieves a chunk about "Hohmann transfer orbits" even though the words don't overlap at all — because the *meaning* is close. This was verified directly during development (see commit history).

---

### Stage C — Answer Generation

**What it does:** Combines the retrieved chunks into one context block, builds a prompt instructing the AI to answer using *only* that context and to admit honestly if the context doesn't contain the answer, and sends it to Groq.

**Why the explicit "say so honestly" instruction matters:** This is the first line of defense against hallucination — before verification even runs. It was directly tested: asking about "the capital of France" (completely outside the stored papers) correctly returned "the context doesn't contain the answer" instead of an invented response.

---

### Stage D — Verification (the differentiator)

**What it does:** Takes the generated answer and the *same* source chunks, and asks the AI a completely separate, narrow question: does every factual claim in the answer trace back to the context? The response is converted into a boolean (`is_supported`).

**Why a separate call, not the same conversation:** If you asked the same AI "was your last answer correct?" in the same context, it tends to just agree with itself. A fresh call with only the raw answer + raw sources, and strict instructions, produces a more genuinely independent check.

**Why `temperature=0` for this call specifically:** the verifier's job is a consistency check, not creative writing — at a nonzero temperature the same answer+context pair could get judged "yes" on one run and "no" on the next. Pinning `temperature=0` makes the yes/no verdict itself deterministic. The verdict is also parsed strictly (`verdict.startswith("yes")`), so a hedge like "yes, mostly, though X isn't stated" is correctly treated as unsupported rather than passing on a loose substring match.

**What "supported" actually means here:** the verifier checks *consistency with the retrieved text*, not real-world factual accuracy. An honest "I don't know" answer correctly passes verification, because it's consistent with context that doesn't contain the answer. This was directly observed during testing and is an intentional, documented scope of what verification means in this system.

---

### Stage E — Conditional Retry Loop

**What it does:** If `is_supported` is `False` and the retry count hasn't hit the cap (2 attempts), the graph loops back to Stage C with the same question, generating a new attempt. If supported, or if the retry cap is reached, the graph ends and returns the current answer.

**Why a retry cap, not unlimited retries:** without a limit, a question the system genuinely can't answer from its data would loop forever. The cap guarantees the system always terminates and returns *something*, even if imperfect — this was a deliberate fix added after building the initial loop, once the infinite-loop risk was identified.

**Why each retry actually widens retrieval:** an earlier version re-ran the exact same `top_k=3` search on every retry — meaning a failed verification triggered an identical generation call with identical context, relying on pure randomness to produce a different (hopefully better-supported) answer. Retries now widen retrieval (`top_k = 3 + 2 × attempt`), so a second attempt genuinely has more source material to draw from, not just another roll of the dice on the same three chunks.

**What happens when the retry cap is hit and the answer is still unsupported:** rather than silently returning the last (possibly hallucinated) draft as if it were a normal answer, the system rewrites it into an explicit "I don't have enough reliably supported information to answer this confidently" message, with the unverified best attempt included for transparency. This directly enforces the project's core design goal — never let the system confidently state something its sources don't support — even in the worst case where verification never succeeds.

---

### Stage F — Topic Summarization (`/summarize_topic`, a separate feature from `/ask`)

**What it does:** Given a topic, fetches the most recent arXiv papers on it (newest-first, unlike the relevance-sorted retrieval used elsewhere), summarizes each paper individually in 2-3 sentences, then writes one combined overview across all of them.

**Why this is separate from the `/ask` pipeline:** `/ask` answers a specific question from previously *ingested* chunks in the vector store. `/summarize_topic` is a standalone "give me the current state of research on X" tool — it fetches and summarizes papers live, on demand, without touching Chroma at all.

**Why it explicitly checks for unrelated papers:** the overview prompt is instructed to say so honestly if the fetched papers aren't closely related to each other, rather than forcing artificial connections between unrelated results — the same "don't fabricate coherence" principle behind the rest of ASTRA's design.

---

### Stage G — Citation Attribution

**What it does:** Once an answer is verified as supported, one more LLM call breaks the answer into its individual claims and maps each claim to the specific context chunk(s) (and therefore paper/title/URL) that back it up — returned as `citations` in the `/ask` response, alongside the existing flat `sources` list.

**Why this is a separate call after verification, not part of it:** verification only needs a yes/no judgment; asking for a structured claim-by-claim breakdown at the same time would make that prompt heavier and slower for the common case. Splitting it out means the extra cost is paid once, only for answers that already passed verification — a hallucinated/unsupported answer (see Stage E) skips citation entirely, since attributing claims in a discarded draft isn't useful.

**Why this doesn't block on a rigid schema:** the LLM is asked for a JSON array and the response is parsed defensively — if it wraps the array in prose, extra whitespace, or a code fence, the parser extracts just the `[...]` portion before calling `json.loads`. If parsing still fails, `citations` comes back as an empty list rather than crashing the request; `sources` (the coarser, always-available list) is unaffected either way.

**Tradeoff:** this adds one more LLM call to every successful `/ask` request, on top of the existing generate (+ retries) and verify calls — a deliberate cost/latency-for-transparency tradeoff.

---

## 🔧 Every Function Explained

### `embeddings.py`

#### `get_embedding(text: str) → list[float]`
**What:** Converts any string into a list of ~384 numbers representing its meaning, using the `bge-small-en-v1.5` model running locally.

**Why BGE over MiniLM:** BGE was trained specifically for retrieval tasks (matching questions to relevant passages), which is exactly what this project does — general-purpose embedding models are slightly less precise for this specific job.

---

### `vectorstore.py`

#### `add_chunk(chunk_id: str, text: str, metadata: dict) → None`
**What:** Embeds a piece of text and stores it in Chroma alongside its original text and metadata (paper title, URL, chunk index).

**Why `chunk_id` must be unique:** Chroma uses it to identify each entry — reusing an ID would overwrite previous data instead of adding new data.

#### `search(query: str, top_k: int = 3) → dict`
**What:** Embeds the query and asks Chroma for the `top_k` closest stored chunks by embedding distance.

**Why `top_k` as a parameter with a default:** callers can request more or fewer results without changing the function itself — `search("question")` uses 3 by default, `search("question", top_k=5)` overrides it.

---

### `ingestion.py`

#### `fetch_papers(query: str, max_results: int = 5) → list`
**What:** Calls the arXiv API and returns paper objects (title, abstract, URL, published date) matching the query, sorted by relevance.

**Why relevance sort, not date sort:** date-sorted results returned topically unrelated papers during testing (e.g., a "hypersonic propulsion" search returning a humanoid-robot paper) since arXiv's date sort ignores topical fit entirely.

**Known behavior, not a bug:** arXiv's search is keyword-based, not semantic — so results can be loosely related. This is expected; ASTRA's own embedding-based `search()` provides the actual precision layer on top of these candidate papers.

#### `chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) → list[str]`
**What:** Splits text into overlapping word-count chunks. See Stage A above for why chunking and overlap matter.

#### `ingest_papers(query: str, max_results: int = 5) → int`
**What:** The full pipeline function — calls `fetch_papers()`, then `chunk_text()` on each paper's abstract, then `add_chunk()` for every resulting chunk. Returns the total number of chunks saved.

**Why unique chunk IDs use `{arxiv_id}_chunk_{index}`:** guarantees no collision between chunks from different papers or different positions within the same paper.

#### `summarize_paper(paper) → dict`
**What:** Summarizes a single paper's abstract in 2-3 sentences via an LLM call, and returns it alongside the paper's title, authors, publication date, and URL.

#### `summarize_topic(topic: str, max_results: int = 5) → dict`
**What:** Fetches the most recent papers on a topic (`sort_by_date=True`), runs `summarize_paper()` on each, then makes one more LLM call to produce a short combined overview across all the individual summaries. Returns the topic, the list of per-paper summaries, and the overall overview.

---

### `llm.py`

#### `ask_ai(question: str) → str`
**What:** Sends a single message to Groq's `openai/gpt-oss-120b` model and returns the text response.

**Why this is a separate file from `rag.py`/`graph.py`:** keeps the raw "talk to the LLM" logic isolated from the RAG-specific logic (prompt building, context injection) — a clean separation of concerns.

---

### `graph.py`

#### `generate_node(state) → state`
**What:** Runs Stage B (retrieval) and Stage C (generation), storing the retrieved chunks and the draft answer into the graph's shared state. Also increments `loop_count`.

#### `verify_node(state) → state`
**What:** Runs Stage D — checks the draft answer against the stored context chunks and sets `state["is_supported"]`.

#### `decide_next_step(state) → str`
**What:** The conditional logic — returns `"end"` if supported or the retry cap is reached, otherwise `"retry"`. This return value is used by LangGraph's `add_conditional_edges` to decide which node runs next. Note that `"end"` routes to the `cite` node, not straight to `END` — see below.

**Why this is a plain function, not a node:** it doesn't transform the state, it only makes a routing decision — LangGraph treats these as a distinct concept (a "conditional edge") from a "node."

#### `cite_node(state) → state`
**What:** Runs Stage G — for a verified answer, asks the LLM to map each claim in the answer to the context chunk(s) that support it, and stores the result (with title/URL attached) as `state["citations"]`. For an answer that never passed verification, sets `citations` to an empty list without making the extra call.

---

### `main.py`

#### `POST /ingest`
**What:** Accepts `{"topic": str, "max_results": int}`, calls `ingest_papers()`, returns how many chunks were saved.

#### `POST /ask`
**What:** Accepts `{"query": str}`, invokes the compiled LangGraph pipeline, and returns the final answer along with `supported` (bool), `attempts` (loop count), `sources` (flat, deduplicated list of retrieved papers), and `citations` (claim-by-claim attribution — empty list if the answer never passed verification) — deliberately exposing this metadata so API consumers can see whether an answer was verified, not just trust it blindly.

#### `POST /summarize_topic`
**What:** Accepts `{"topic": str, "max_results": int}`, calls `summarize_topic()`, and returns `{"topic": str, "papers": [...], "overall_summary": str}` — a live, on-demand literature-review-style overview of a topic, independent of anything previously ingested.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/BiswasNehaa/Astra.git
cd Astra
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a `.env` file:
GROQ_API_KEY=your_groq_api_key_here

Get a free key at [console.groq.com](https://console.groq.com)

### 5. Run Locally
```bash
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/docs` to test endpoints interactively, or `http://127.0.0.1:8000/ui/` for a simple demo page to ingest a topic and ask questions from the browser.

### 6. Run with Docker
```bash
docker build -t astra .
docker run -p 8000:8000 --env-file .env astra
```

### 7. Run Tests
```bash
pip install -r requirements-dev.txt
pytest
```
Tests cover the pure logic (`chunk_text`, the retry-loop's `decide_next_step`, and `generate_node`/`verify_node`'s contracts) with `vectorstore`/`llm` stubbed out, so they run fast without needing a Groq API key, a downloaded embedding model, or network access.

---

## 📥 Example Input & Output

**Step 1 — Ingest papers on a topic:**
```json
POST /ingest
{"topic": "quantum computing", "max_results": 5}
```
```json
{"chunks_saved": 7}
```

**Step 2 — Ask a question:**
```json
POST /ask
{"query": "How do you move a spacecraft between two orbits?"}
```
```json
{
  "answer": "A spacecraft can be moved between two circular orbits using a Hohmann transfer orbit, which involves two engine burns.",
  "supported": true,
  "attempts": 1,
  "sources": [
    {"title": "Optimal Orbital Transfer Strategies", "url": "https://arxiv.org/abs/..."}
  ],
  "citations": [
    {
      "claim": "A Hohmann transfer orbit moves a spacecraft between two circular orbits.",
      "sources": [
        {"title": "Optimal Orbital Transfer Strategies", "url": "https://arxiv.org/abs/..."}
      ]
    },
    {
      "claim": "It involves two engine burns.",
      "sources": [
        {"title": "Optimal Orbital Transfer Strategies", "url": "https://arxiv.org/abs/..."}
      ]
    }
  ]
}
```

**Step 3 — Get a live overview of recent research on a topic (no ingestion needed):**
```json
POST /summarize_topic
{"topic": "quantum error correction", "max_results": 3}
```
```json
{
  "topic": "quantum error correction",
  "papers": [
    {
      "title": "...",
      "authors": ["..."],
      "published": "2026-01-15",
      "url": "https://arxiv.org/abs/...",
      "summary": "..."
    }
  ],
  "overall_summary": "..."
}
```

---

## 🤔 Design Decisions & Tradeoffs

| Decision | Alternative Considered | Why This Choice Won |
|---|---|---|
| Chroma over Qdrant/FAISS | Qdrant (separate server), FAISS (lower-level) | Built-in persistence and metadata filtering, far less setup code for a project this size |
| BGE over MiniLM embeddings | `all-MiniLM-L6-v2` | BGE is trained specifically for retrieval tasks, a better fit for RAG specifically |
| Groq over OpenAI/Claude API | Paid APIs | Free tier, fast inference, zero cost pressure while learning and iterating |
| Verification as a separate LLM call | Self-critique in the same conversation | A fresh call with only raw inputs avoids the AI simply agreeing with its own prior output |
| Retry cap of 2 | Unlimited retries | Guarantees termination; prevents infinite loops on genuinely unanswerable questions |
| **Abstracts only, not full PDF text** | Parsing full PDF text | Abstracts are always available via the arXiv API with zero parsing complexity or failure modes; sufficient for a "research discovery and comparison" use case. **This is a real scope boundary, documented honestly below, not a hidden gap.** |
| Python 3.11 in Docker, not 3.13 | Matching local Python version | `chroma-hnswlib` and other packages lacked pre-built installers for 3.13 at the time, forcing slow/broken source compilation; 3.11 has full pre-built wheel support |
| CPU-only PyTorch in Docker | Default PyTorch install | Default install pulls multi-GB NVIDIA CUDA libraries never used in this containerized, GPU-less deployment — CPU-only build cut image size dramatically and fixed unstable, timing-out builds |

---

## ⚠️ Known Limitations

- **Abstracts only, not full papers.** Ingestion pulls arXiv abstracts (150-250 words), not full PDF text. ASTRA can answer questions about a paper's main claims, methods at a high level, and general findings — but not fine-grained details (exact numbers, full methodology, limitations sections) that only appear in the full paper body. Full-text PDF ingestion is tracked as an open contribution issue.
- **Verification checks consistency, not ground truth.** The verifier confirms an answer matches the retrieved context — it does not independently confirm the retrieved context itself is factually correct. An honestly-worded "I don't know" answer correctly passes verification. If the retrieved chunks themselves are wrong or off-topic (e.g. `top_k` widening on retry still surfaces irrelevant chunks because nothing relevant exists in the vector store for that topic), the answer can be "supported" by bad context — ingest more/better papers on the topic first if answers seem consistently off.
- **Free-tier deployment is memory-constrained.** Local embedding model + PyTorch typically exceed the 512MB RAM limit on Render's free tier. Fully verified working locally and in Docker; production deployment would need a paid instance or a hosted embeddings API instead of a locally-run model.
- **No per-user rate limiting yet** on the `/ask` endpoint — relies on Groq's own free-tier limits.
- **arXiv's own search is keyword-based, not semantic** — candidate papers pulled by `fetch_papers()` can be loosely topically related; ASTRA's own embedding search provides the actual precision layer on top.
- **Sources list shows all retrieved chunks, not just ones actually used in the answer** — may include loosely-related or unlabeled (test data) entries.

---

## 🌟 Future Enhancements

- **Full PDF ingestion** — parse complete paper text, not just abstracts, for deeper Q&A
- **Hosted embeddings API** — remove the local model's memory footprint, enabling free-tier deployment without hitting RAM limits
- **Per-user rate limiting** on `/ask`
- **Multi-paper comparison mode** — "compare the approach in paper A vs paper B"
- **Persistent, larger-scale vector storage** — migrate from Chroma to Qdrant for larger paper corpora
- **CI workflow** — run the test suite automatically on every push/PR

---

## 🔐 Security Notes

- API keys are stored in `.env` and never hardcoded or committed to git (verified via `.gitignore` and confirmed with `git ls-files` during development)
- `.dockerignore` excludes `.venv`, `.git`, and local database files from the Docker build context

---

*Built with FastAPI · LangGraph · Groq (`openai/gpt-oss-120b`) · sentence-transformers (BGE) · ChromaDB · Docker*
