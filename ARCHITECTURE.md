Here’s a compact, production-ready blueprint you can build locally with Python + Quart, an open-source LLM (via Ollama), and a RAG layer over your `notes/*.md`. I’ve kept the UI lightweight (Tailwind + DaisyUI, minimal JS, no npm at runtime) and included an MCP-style tool layer for weather + web search.

# 1) High-level architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                               Browser UI                                 │
│  - Quart templates (Jinja2), Tailwind+DaisyUI (precompiled CSS)          │
│  - Chat page, Notes browser, Settings                                    │
└───────────────▲───────────────────────────────────────────────────────────┘
                │ HTTP (WebSocket optional for streaming)
┌───────────────┴───────────────────────┐
│               Quart API               │
│  /chat  /retrieve  /tools  /notes     │
│  - Session auth (optional)            │
│  - SSE or WS for token streaming      │
└───────────────┬───────────────────────┘
                │
                │ orchestrates
┌───────────────┴───────────────────────────────────────────────────────────┐
│                   Orchestrator / Chat Engine                              │
│  - Conversation Manager (memory policy)                                   │
│  - RAG Pipeline (ingest → embed → store → retrieve)                       │
│  - Tool Router (MCP-style tools: weather, web search, etc.)               │
│  - LLM Client (Ollama)                                                    │
└───────────────┬───────────────────────────────────────────────────────────┘
                │                            │
        uses    │                            │ calls/tools
                │                            ▼
        ┌───────▼───────────┐        ┌───────────────────────┐
        │ Embedding Models  │        │   MCP Tools Registry  │
        │ - Ollama (e.g.,   │        │ - weather.get(...)    │
        │   nomic-embed,    │        │ - web.search(...)     │
        │   mxbai-embed)    │        │ - (extensible)        │
        └───────┬───────────┘        └──────────┬────────────┘
                │                                 │
                │ writes/read vectors             │ external HTTP where needed
┌───────────────▼─────────────────────────────────▼──────────────────────────┐
│                             Local Storage                                  │
│  SQLite (metadata + chat logs)    +    FAISS (or Chroma/LanceDB/pgvector)  │
│  - tables: sessions, messages, notes_index, chunks, tool_runs              │
│  - vector store: chunk embeddings                                          │
└────────────────────────────────────────────────────────────────────────────┘

Ingest sidecar (watcher):
- Monitors notes/ for .md changes → chunk → embed → upsert vectors & metadata
```

# 2) Recommended tooling & libraries

**Core**

* **Quart** (async Flask) for API/UI: `quart`, `hypercorn`.
* **Ollama** for local LLM & embeddings: `ollama` Python client.
* **RAG**

  * **Chunking**: `langchain-text-splitters` (Markdown splitter) or `llama-index-core` splitters.
  * **Vector store**: **FAISS** (simple, file-based), or **Chroma** (local serverless), or **LanceDB** (great local DB), or **pgvector** if you prefer Postgres.
  * **Embeddings**: **Ollama embeddings** (`nomic-embed-text`, `mxbai-embed-large`), or `sentence-transformers` (e.g., `bge-small-en`) if you want pure Python.
* **MCP-style tools**

  * Write a **Tool Registry** with Pydantic schemas (input/output), callable from the LLM loop.
  * Weather: small client for **Open-Meteo** (no key) or another local-friendly API.
  * Web search: wrapper around **SerpAPI-like** or **DuckDuckGo** HTTP (choose one that works keyless or with a local binary).
* **Persistence**

  * **SQLite** (messages/sessions/metadata).
  * **FAISS** (vectors) on disk alongside SQLite; or use **Chroma** embedded mode; or **pgvector** if you want SQL-native vectors.
* **UI**

  * **Tailwind CLI standalone** to precompile CSS (no npm at runtime).
  * **DaisyUI** compiled into your CSS during a one-time build step (ship the generated CSS with the app).
  * Minimal Alpine.js for progressive enhancement (optional).

# 3) RAG pipeline (end-to-end)

**Ingest (one-time + continuous via watcher)**

1. **Watch**: `watchdog` monitors `notes/*.md` (create/update/delete).
2. **Parse**: read Markdown; extract optional frontmatter (title, tags, dates), normalize paths & timestamps.
3. **Chunk**: Markdown-aware splitter with hierarchy retention (H1/H2/H3) and overlaps (e.g., 600 tokens, 80 overlap).
4. **Embed**: use **Ollama embeddings** (`embeddings(model="nomic-embed-text")`) → 768/1024-d vectors.
5. **Upsert**:

   * Write chunk metadata to SQLite: `note_id`, `chunk_id`, `rel_path`, `headings_path`, `token_count`, `updated_at`.
   * Upsert vectors + `chunk_id` into FAISS (or your chosen vector DB).
6. **Delete handling**: on file deletion, remove its chunks and vectors.

**Query (per user message)**

1. **Question rewrite** (optional): LLM rewrites user query using short-term memory for clarity.
2. **Hybrid retrieval**:

   * Vector search top-k (e.g., 8–12) with **MMR** re-ranking.
   * Optional keyword filter (SQLite FTS5 on chunks) to tighten scope.
3. **Context packing**:

   * Deduplicate by note.
   * Limit total tokens (e.g., 3–5 chunks max ~ 1–1.5k tokens).
   * Include metadata (title, section heading, path).
4. **Answer synthesis**:

   * System prompt enforces grounded answers.
   * Provide citations (note path + heading) in the response.
5. **(Optional) Iterative refinement**:

   * If uncertainty high, do a second, narrower retrieval pass.

# 4) Conversational memory strategy

**Stores (SQLite)**

* `sessions(id, created_at, title, metadata)`
* `messages(id, session_id, role, content, tokens, created_at)`
* `summaries(id, session_id, content, upto_message_id, created_at)` — rolling summaries

**Policy**

* **Short-term memory**: last N user+assistant turns (token-limited window).
* **Long-term memory**: periodic **conversation summary** every ~20–30 messages (LLM distilled), appended to the system context for that session.
* **RAG memory** (optional): embed **message snippets** with tags and store in the same vector DB under a separate namespace for “episodic memory”; retrieve alongside notes with a lower weight.
* **Privacy**: all local; enable per-session “forget last answer” and “clear session.”

# 5) MCP-style tool calling (weather + search)

Because many local LLMs don’t have native function-calling, implement a **tool loop** that:

1. Prompts the model to **respond in JSON** when it wants a tool, with a fixed schema:

   ```json
   { "action": "call_tool", "tool": "weather.get", "args": {"lat": 38.72, "lon": -9.14} }
   ```
2. Validate with **Pydantic**; execute tool; return a **tool_result** message to the model; ask it to produce a final answer.
3. If the model replies with `"action": "final"`, render that to the user.

**Tool Registry (Python)**

```python
@dataclass
class Tool:
    name: str
    description: str
    input_model: type[pydantic.BaseModel]
    output_model: type[pydantic.BaseModel]
    handler: Callable[[BaseModel], BaseModel]

TOOLS = {
  "weather.get": Tool(...),
  "web.search": Tool(...),
}
```

**Weather tool**: call Open-Meteo (free) and return a concise struct (current temp, condition, next 12h).
**Web search tool**: DuckDuckGo lite client (HTML scraping or API) → titles + urls + snippets.

This mirrors **MCP** concepts (declarative tools with schemas); if you later adopt a full MCP host/runtime, your registry maps cleanly.

# 6) LLM invocation patterns (Ollama)

* **Chat model**: `llama3.1`, `mistral-nemo`, or `qwen2.5` sized to your hardware.
* **Embeddings**: `nomic-embed-text` or `mxbai-embed-large`.
* **Streaming**: enable token streams over SSE or WS.
* **Guardrails**: use structured output (e.g., `instructor` or `pydantic` parser) only when in tool mode; otherwise plain text.

**System prompt skeleton**

* Goals: answer using **only** retrieved notes unless user asks for general knowledge; cite sections; if missing info, say so and optionally suggest the web search tool.
* Tool mode: “If a tool is helpful, respond with JSON per schema; otherwise respond with natural language.”

# 7) Minimal API surface (Quart)

```
POST /api/chat        -> {session_id, message}   # handles tool loop + RAG + memory
GET  /api/sessions    -> list sessions
POST /api/sessions    -> create session
GET  /api/messages    -> {session_id} -> messages
GET  /api/notes       -> browse indexed notes (path, titles, updated_at)
POST /api/reindex     -> trigger ingest (or run watcher as a sidecar)
GET  /stream/chat     -> SSE/WS stream of tokens (optional)
```

# 8) File/folder scaffold (no runtime npm)

```
ai-assistant/
├─ app/
│  ├─ main.py                # Quart app factory, routes, SSE/WS
│  ├─ llm_client.py          # Ollama chat & embeddings
│  ├─ rag/
│  │  ├─ ingest.py           # watchdog loop, parse/chunk/embed/upsert
│  │  ├─ retriever.py        # vector search (FAISS/Chroma), MMR, hybrid
│  │  ├─ store_faiss.py      # vector index wrapper
│  │  └─ md_parser.py        # frontmatter + markdown utils
│  ├─ tools/
│  │  ├─ registry.py         # MCP-style registry
│  │  ├─ weather.py          # weather.get
│  │  └─ websearch.py        # web.search
│  ├─ memory/
│  │  ├─ manager.py          # short/long-term policy, summaries
│  │  └─ schemas.py          # pydantic models for messages/sessions
│  ├─ db.py                  # SQLite init, DAL
│  ├─ prompts/
│  │  ├─ system_base.txt
│  │  └─ tool_instruction.txt
│  └─ config.py              # paths, model names, chunk sizes, top_k, etc.
├─ notes/                    # your .md files (watched)
├─ data/
│  ├─ vectors.index          # FAISS files
│  └─ assistant.sqlite       # SQLite DB
├─ web/
│  ├─ templates/             # Jinja2 templates (chat.html, layout.html)
│  └─ static/
│     ├─ css/app.css         # precompiled Tailwind + DaisyUI
│     └─ js/app.js           # tiny Alpine.js (optional)
├─ scripts/
│  ├─ build_css.sh           # tailwind standalone build step (once)
│  └─ reindex.py             # manual full reindex
└─ pyproject.toml / requirements.txt
```

> **Tailwind + DaisyUI without npm at runtime:** run the Tailwind CLI (standalone binary) with DaisyUI plugin **during a one-time build step**, commit `web/static/css/app.css` to the repo, and serve it statically. The app itself needs no Node/npm running.

# 9) Key implementation notes

**Chunking**

* Prefer Markdown splitter that respects headings; capture a `heading_path` like `# Project → ## Setup → ### DB`.
* Overlap to preserve cross-paragraph references (e.g., 80–100 tokens).

**Retrieval quality**

* Use **MMR** to reduce redundancy.
* Consider **hybrid**: vector top-k ∩ keyword FTS filter in SQLite.
* Cap total context tokens (e.g., ~1500) to keep small local models snappy.

**Citations**

* Include `[note_path#heading]` anchors in the final answer.
* Don’t hallucinate: if confidence is low, propose `web.search` tool.

**Tool loop**

* Timebox tools and sanitize outputs.
* Log tool calls in `tool_runs` table for auditability.

**Memory**

* Summarize older turns (“conversation TL;DR”) every N messages.
* Keep per-session summary in `summaries` and feed it as compressed context before the last K turns.
* Allow per-session **“pin this fact”** → store as small memory snippets (optionally vectorized) retrievable like notes with a lower weight.

# 10) Example flows

**A) User asks a question about notes**

1. `POST /api/chat`
2. Conversation Manager selects: system prompt + summary + last 6 turns.
3. RAG retrieves chunks (top-k, MMR).
4. LLM composes grounded answer with citations.
5. Message stored; optional rolling summary updated.

**B) User asks: “What’s the weather in Lisbon?”**

1. Model decides tool needed → returns `{"action":"call_tool","tool":"weather.get","args":{"lat":38.72,"lon":-9.14}}`.
2. Tool Registry executes, returns structure.
3. Model produces final natural-language reply.

**C) User says: “Search the web for X”**

1. Same tool path via `web.search`, return top results.
2. Optionally retrieve pages and run mini-RAG across fetched snippets (local only).

# 11) Sensible defaults

* **Models** (Ollama):

  * Chat: `llama3.1:8b` (fast) → upgrade to `70b` if you can.
  * Embeddings: `nomic-embed-text` (fast, solid).
* **RAG params**: chunk 600 tokens / overlap 80; top-k=8; MMR λ=0.5; max context ~1500 tokens.
* **Stores**: SQLite for everything non-vector; FAISS for vectors (or Chroma embedded mode).
* **UI**: server-rendered Jinja2, 1 small HTMX/Alpine sprinkle for streaming.
