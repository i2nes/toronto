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

# 11) Security & Privacy

**Core principle**: This is a **local-first** system—all data processing, storage, and LLM inference happen on your machine. No data leaves your environment unless you explicitly enable external tools (web search, weather API).

**Authentication & Authorization**

* **Local deployment**: No auth required by default (single-user, localhost-only).
* **Network exposure**: If binding to `0.0.0.0`, add session-based auth with `quart-auth` or simple API keys.
* **Multi-user** (optional): Add `user_id` to sessions/messages; enforce row-level isolation in queries.

**Input Sanitization**

* **User queries**: Sanitize before passing to LLM or tools—strip control characters, limit length (e.g., 2000 chars).
* **File paths**: When indexing notes, validate paths are within `notes/` directory (prevent path traversal).
* **Tool inputs**: Use Pydantic validation on all tool arguments; reject malformed data.

```python
from pydantic import BaseModel, Field, validator

class WeatherRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)

    @validator('lat', 'lon')
    def validate_coords(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Coordinates must be numeric')
        return v
```

**Tool Execution Safety**

* **Sandboxing**: Run tool handlers in `asyncio.timeout()` contexts; terminate after 30s max.
* **Resource limits**: For web search, cap results (e.g., max 10); for weather, cache responses (5min TTL).
* **Logging**: Log all tool calls with args/results to `tool_runs` table for audit trail.
* **Deny list**: Maintain list of prohibited tool actions (e.g., file write, shell exec) unless explicitly enabled.

```python
async def execute_tool(tool_name: str, args: BaseModel) -> ToolResult:
    """Execute tool with timeout and error handling."""
    if tool_name not in ALLOWED_TOOLS:
        return ToolResult(success=False, error="Tool not permitted")

    try:
        async with asyncio.timeout(30):
            handler = TOOLS[tool_name].handler
            result = await handler(args)
            await log_tool_run(tool_name, args, result, success=True)
            return ToolResult(success=True, data=result)
    except asyncio.TimeoutError:
        await log_tool_run(tool_name, args, None, success=False, error="timeout")
        return ToolResult(success=False, error="Tool execution timeout")
    except Exception as e:
        logger.exception(f"Tool {tool_name} failed")
        await log_tool_run(tool_name, args, None, success=False, error=str(e))
        return ToolResult(success=False, error="Tool execution failed")
```

**Secrets Management**

* **Environment variables**: Store API keys in `.env` (never commit).
* **Example file**: Maintain `.env.example` with placeholder values.
* **Runtime**: Load with `python-dotenv`; validate required keys on startup.

```bash
# .env.example
WEATHER_API_KEY=your_key_here
SEARCH_API_KEY=optional_if_using_serpapi
OLLAMA_BASE_URL=http://localhost:11434
```

**Data Privacy**

* **Local storage only**: SQLite + FAISS files in `data/` directory.
* **Session cleanup**: Provide `/api/sessions/<id>/delete` endpoint; cascade-delete messages, summaries, tool_runs.
* **Note exclusion**: Allow `.rag-ignore` file to exclude sensitive notes from indexing.
* **Encryption at rest** (optional): Use SQLCipher for SQLite encryption if handling sensitive notes.

**Prompt Injection Mitigation**

* **System prompt isolation**: Keep system instructions in a separate message; never let user input overwrite system context.
* **Tool call validation**: Parse LLM JSON output strictly; reject if schema doesn't match exactly.
* **Output filtering**: Scan assistant responses for leaked system prompts or internal paths before displaying.

**Dependencies & Supply Chain**

* **Pin versions**: Use `requirements.txt` with exact versions (`==` not `>=`).
* **Audit**: Run `pip-audit` regularly to check for CVEs.
* **Minimal surface**: Avoid heavy frameworks; prefer stdlib + targeted libs (Quart, Ollama client, FAISS).

# 12) Error Handling & Resilience

**LLM Client Failures**

* **Ollama unavailable**: Check `http://localhost:11434/api/tags` on startup; fail fast with clear error.
* **Retry logic**: Retry transient failures (connection errors) with exponential backoff (max 3 attempts).
* **Timeout**: Set `timeout` on Ollama API calls (e.g., 60s for chat, 30s for embeddings).
* **Graceful degradation**: If LLM fails mid-conversation, return cached "Service temporarily unavailable" message; preserve session state.

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_ollama_chat(messages: list[dict]) -> str:
    try:
        response = await ollama_client.chat(
            model="llama3.1:8b",
            messages=messages,
            timeout=60.0
        )
        return response['message']['content']
    except httpx.TimeoutException:
        logger.error("Ollama chat timeout")
        raise
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama")
        raise
```

**RAG Pipeline Errors**

* **Empty retrieval**: If no chunks found, return fallback response: *"I couldn't find relevant notes. Would you like me to search the web?"*
* **Embedding failures**: Log error, skip chunk, continue indexing (don't fail entire ingest).
* **Vector store corruption**: Keep daily backups of FAISS index; auto-restore or trigger full reindex.
* **File watcher crashes**: Wrap watchdog in supervisor; restart on unhandled exceptions; log all events.

**Tool Execution Failures**

* **Network errors** (weather/search): Return partial result or error message; don't crash the conversation.
* **Malformed tool calls**: If LLM outputs invalid JSON, log warning, ask LLM to retry with a corrected prompt.
* **Rate limiting**: Implement exponential backoff for external APIs; cache results aggressively.

```python
async def weather_tool(args: WeatherRequest) -> WeatherResult:
    cache_key = f"weather:{args.lat}:{args.lon}"
    cached = await get_cache(cache_key)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": args.lat, "longitude": args.lon, "current_weather": True}
            )
            response.raise_for_status()
            data = response.json()
            result = WeatherResult.parse_obj(data['current_weather'])
            await set_cache(cache_key, result, ttl=300)  # 5min
            return result
    except httpx.HTTPError as e:
        logger.error(f"Weather API error: {e}")
        return WeatherResult(error="Weather service unavailable")
```

**Database Failures**

* **SQLite locked**: Use WAL mode (`PRAGMA journal_mode=WAL`) for concurrent reads.
* **Disk full**: Check available space before writes; fail gracefully with user-facing error.
* **Schema migrations**: Use Alembic for versioned migrations; test rollback paths.

**Conversation Memory Limits**

* **Token overflow**: If conversation exceeds max context (e.g., 8k tokens), force summarization of older turns.
* **Summary failures**: If summary LLM call fails, fall back to truncating oldest messages (keep last N turns).
* **Corruption recovery**: If `messages` table corrupted, allow session export (JSON) and recovery from export.

**Streaming Failures**

* **SSE/WebSocket drops**: Client should auto-reconnect; server keeps partial generation in Redis/memory for resume.
* **Partial generation**: If stream interrupted, store partial response with `completed=false` flag; allow retry.

# 13) Observability & Monitoring

**Logging Strategy**

* **Structured logs**: Use `structlog` for JSON-formatted logs with context (session_id, user_id, request_id).
* **Log levels**:
  * `DEBUG`: LLM prompts, tool args, retrieval queries.
  * `INFO`: API requests, session lifecycle, ingest events.
  * `WARNING`: Retry attempts, partial failures, slow queries (>1s).
  * `ERROR`: LLM failures, tool crashes, DB errors.
* **Log rotation**: Use `logging.handlers.RotatingFileHandler` (max 50MB per file, keep 5 backups).

```python
import structlog

logger = structlog.get_logger()

# Usage
await logger.ainfo(
    "chat_message_processed",
    session_id=session.id,
    message_tokens=len(tokens),
    retrieval_chunks=len(chunks),
    latency_ms=elapsed * 1000
)
```

**Metrics to Track**

* **LLM metrics**:
  * Requests per minute.
  * Average latency (p50, p95, p99).
  * Token usage (input/output per request).
  * Error rate (by error type).
* **RAG metrics**:
  * Ingest rate (chunks/sec).
  * Retrieval latency.
  * Cache hit rate (for repeated queries).
  * Average chunks per query.
* **Tool metrics**:
  * Tool call frequency (by tool name).
  * Success/failure rate.
  * Average execution time.
* **API metrics**:
  * Request rate (by endpoint).
  * Response time.
  * 4xx/5xx error rate.

**Implementation Options**

* **Simple (local)**: Log metrics to SQLite `metrics` table; build dashboards with Grafana + SQLite datasource.
* **Prometheus** (optional): Expose `/metrics` endpoint with `prometheus-client`; scrape with local Prometheus + Grafana.
* **OpenTelemetry** (future): Add OTLP exporter for distributed tracing if scaling beyond single-user.

```python
from prometheus_client import Counter, Histogram, generate_latest

chat_requests = Counter('chat_requests_total', 'Total chat requests', ['status'])
chat_latency = Histogram('chat_latency_seconds', 'Chat request latency')

@app.route('/metrics')
async def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}

@app.route('/api/chat', methods=['POST'])
@chat_latency.time()
async def chat():
    try:
        result = await process_chat(request.json)
        chat_requests.labels(status='success').inc()
        return jsonify(result)
    except Exception as e:
        chat_requests.labels(status='error').inc()
        raise
```

**Health Checks**

* **Readiness probe**: `GET /health/ready` → check Ollama connectivity, DB reachable, FAISS index loaded.
* **Liveness probe**: `GET /health/live` → return 200 if app is running (no deps checked).

```python
@app.route('/health/ready')
async def health_ready():
    checks = {
        'ollama': await check_ollama(),
        'database': await check_database(),
        'vector_store': await check_vector_store()
    }
    if all(checks.values()):
        return jsonify(checks), 200
    else:
        return jsonify(checks), 503

async def check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get('http://localhost:11434/api/tags')
            return resp.status_code == 200
    except Exception:
        return False
```

**Debugging Tools**

* **Query inspector**: Add `/api/debug/last-retrieval` endpoint to show chunks, scores, reranking for last query (dev only).
* **Conversation replay**: Export session as JSON; replay through LLM with modified prompts for testing.
* **Prompt viewer**: Show effective prompt (system + summary + context + user message) in debug mode.

**Performance Profiling**

* **Slowlog**: Log any request >2s with full trace (retrieval time, LLM time, tool time).
* **cProfile**: Use `cProfile` or `py-spy` to profile slow endpoints.
* **FAISS index stats**: Track index size, query time as notes grow.

**Alerting** (optional for production-like setups)

* **Error rate spike**: Alert if error rate >5% over 5min window.
* **Ollama down**: Alert if health check fails 3 times consecutively.
* **Disk space low**: Alert if `data/` partition <10% free.

**Example Dashboard Panels** (Grafana/Metabase)

1. **Chat volume**: Line chart of requests/min.
2. **Latency**: Histogram (p50/p95/p99).
3. **RAG quality**: Average chunks retrieved, cache hit rate.
4. **Tool usage**: Pie chart of tool calls by type.
5. **Error log**: Table of recent errors with stack traces.

# 14) Sensible defaults

* **Models** (Ollama):

  * Chat: `gemma3:12b`
  * Embeddings: `mxbai-embed-large` (fast, solid).
* **RAG params**: chunk 600 tokens / overlap 80; top-k=8; MMR λ=0.5; max context ~1500 tokens.
* **Stores**: SQLite for everything non-vector; FAISS for vectors (or Chroma embedded mode).
* **UI**: server-rendered Jinja2, 1 small HTMX/Alpine sprinkle for streaming.
