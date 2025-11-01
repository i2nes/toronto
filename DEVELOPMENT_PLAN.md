# Development Plan: AI Assistant with RAG

**Goal:** Build incrementally with early UI feedback. Each phase delivers working, testable software.

**Philosophy:**
- Get UI running ASAP (Phase 1 = basic chat)
- Add complexity incrementally (RAG → Tools → Memory)
- Test continuously at each milestone
- Small PRs, clear commits (following CLAUDE.md)

---

## 🎯 Current Status

```
✅ Phase 0: Project Bootstrap (COMPLETED - 2025-11-01)
✅ Phase 1: Basic Chat UI (COMPLETED - 2025-11-01)
✅ Phase 2: RAG Pipeline - Indexing (COMPLETED - 2025-11-01)
✅ Phase 3: RAG-Enhanced Chat (COMPLETED - 2025-11-01)
📋 Phase 4: Sessions & Memory (PENDING)
📋 Phase 5: Tool Calling (PENDING)
📋 Phase 6: Polish & Production (PENDING)
📋 Phase 7: Advanced Features (OPTIONAL)
```

**🎉 Working Demo:** http://localhost:5001 (RAG-enhanced chat is live!)

**⏱️ Time Spent:**
- Phase 0: ~2 hours (setup, validation)
- Phase 1: ~2 hours (implementation + Alpine.js refactor)
- Phase 2: ~2 hours (RAG pipeline implementation)
- Phase 3: ~2 hours (retrieval integration + UI updates)
- **Total:** ~8 hours from zero to RAG-enhanced chat

**📦 What's Built:**
- ✅ Full-stack Quart app with async Ollama client
- ✅ Alpine.js-powered chat interface (minimal JS)
- ✅ Tailwind + DaisyUI styling
- ✅ Health checks and error handling
- ✅ Validation script for setup verification
- ✅ Complete RAG indexing pipeline (markdown → chunks → embeddings → FAISS)
- ✅ 11 chunks indexed from 3 sample notes
- ✅ Runtime embedding dimension detection (1024d)
- ✅ Semantic retrieval with FAISS vector search
- ✅ Context-enhanced chat responses
- ✅ Source citations in UI (collapsible sources with relevance scores)

---

## Phase 0: Project Bootstrap ✅ COMPLETED

**Goal:** Scaffold the project, install dependencies, validate Ollama setup.

**Status:** ✅ Completed 2025-11-01

### Tasks

1. **Project structure**
   ```bash
   mkdir -p app/{rag,tools,memory,prompts} web/{templates,static/css,static/js} data notes scripts tests
   touch app/{main.py,llm_client.py,db.py,config.py}
   touch app/rag/{ingest.py,retriever.py,store_faiss.py,md_parser.py}
   touch app/tools/{registry.py,weather.py,websearch.py}
   touch app/memory/{manager.py,schemas.py}
   ```

2. **Dependencies** (`requirements.txt`)
   ```
   # Core
   quart==0.19.4
   hypercorn==0.16.0

   # LLM & embeddings
   ollama==0.1.6
   httpx==0.26.0

   # RAG
   faiss-cpu==1.7.4
   langchain-text-splitters==0.0.1

   # Tools
   pydantic==2.5.3
   python-dotenv==1.0.0

   # DB & utils
   aiosqlite==0.19.0
   watchdog==4.0.0

   # Logging & monitoring
   structlog==24.1.0

   # Testing
   pytest==7.4.3
   pytest-asyncio==0.23.3
   pytest-playwright==0.4.4

   # Dev tools
   black==23.12.1
   isort==5.13.2
   ruff==0.1.9
   ```

3. **Environment setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Ollama validation**
   ```bash
   # Pull required models
   ollama pull mistral-nemo:12b-instruct-q4_K_M
   ollama pull mxbai-embed-large

   # Verify
   curl http://localhost:11434/api/tags
   ```

5. **Config file** (`app/config.py`)
   ```python
   from pathlib import Path

   # Paths
   BASE_DIR = Path(__file__).parent.parent
   DATA_DIR = BASE_DIR / "data"
   NOTES_DIR = BASE_DIR / "notes"

   # Ollama
   OLLAMA_BASE_URL = "http://localhost:11434"
   CHAT_MODEL = "mistral-nemo:12b-instruct-q4_K_M"
   EMBEDDING_MODEL = "mxbai-embed-large"

   # RAG (will use in Phase 2)
   CHUNK_SIZE = 2400
   CHUNK_OVERLAP = 320
   RETRIEVAL_TOP_K = 8
   MMR_LAMBDA = 0.5
   MAX_CONTEXT_TOKENS = 1500

   # UI
   STATIC_VERSION = "1.0.0"
   ```

6. **Tailwind CSS setup**
   ```bash
   # Download Tailwind standalone binary
   curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-macos-arm64
   chmod +x tailwindcss-macos-arm64
   mkdir -p bin
   mv tailwindcss-macos-arm64 bin/tailwindcss

   # Create Tailwind config
   cat > tailwind.config.js <<EOF
   module.exports = {
     content: ["./web/templates/**/*.html"],
     plugins: [require("daisyui")],
   }
   EOF

   # Create input CSS
   mkdir -p web/static/src
   cat > web/static/src/styles.css <<EOF
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   EOF

   # Build CSS (one-time)
   ./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --minify
   ```

**Deliverable:** Project scaffolded, dependencies installed, Ollama validated, CSS built.

---

## Phase 1: Basic Chat UI ✅ COMPLETED

**Goal:** Working chat interface with Ollama (no RAG yet). You can interact with the LLM immediately.

**Status:** ✅ Completed 2025-11-01
**Demo:** http://localhost:5001

### Tasks

1. **LLM client** (`app/llm_client.py`)
   ```python
   import httpx
   from app import config

   class OllamaClient:
       async def chat(self, messages: list[dict], stream: bool = False):
           async with httpx.AsyncClient(timeout=60.0) as client:
               response = await client.post(
                   f"{config.OLLAMA_BASE_URL}/api/chat",
                   json={
                       "model": config.CHAT_MODEL,
                       "messages": messages,
                       "stream": stream
                   }
               )
               response.raise_for_status()
               return response.json()
   ```

2. **Basic Quart app** (`app/main.py`)
   ```python
   from quart import Quart, render_template, request, jsonify
   from app.llm_client import OllamaClient
   from app import config

   app = Quart(__name__, template_folder="../web/templates", static_folder="../web/static")
   llm_client = OllamaClient()

   @app.route("/")
   async def index():
       return await render_template("chat.html", static_version=config.STATIC_VERSION)

   @app.route("/api/chat", methods=["POST"])
   async def chat():
       data = await request.get_json()
       user_message = data.get("message", "")

       messages = [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": user_message}
       ]

       response = await llm_client.chat(messages)
       return jsonify({"response": response["message"]["content"]})

   @app.route("/health/ready")
   async def health():
       return jsonify({"status": "ok"}), 200
   ```

3. **Chat UI template** (`web/templates/chat.html`)
   ```html
   <!DOCTYPE html>
   <html data-theme="light">
   <head>
       <title>AI Assistant</title>
       <link rel="stylesheet" href="/static/css/app.css?v={{ static_version }}">
   </head>
   <body class="bg-base-200">
       <div class="container mx-auto max-w-4xl p-4">
           <div class="card bg-base-100 shadow-xl">
               <div class="card-body">
                   <h2 class="card-title">AI Assistant</h2>

                   <div id="messages" class="space-y-4 h-96 overflow-y-auto mb-4"></div>

                   <div class="form-control">
                       <div class="input-group">
                           <input id="userInput" type="text" placeholder="Ask me anything..."
                                  class="input input-bordered flex-1" />
                           <button onclick="sendMessage()" class="btn btn-primary">Send</button>
                       </div>
                   </div>
               </div>
           </div>
       </div>

       <script>
           async function sendMessage() {
               const input = document.getElementById('userInput');
               const message = input.value.trim();
               if (!message) return;

               addMessage('user', message);
               input.value = '';

               try {
                   const response = await fetch('/api/chat', {
                       method: 'POST',
                       headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify({message})
                   });
                   const data = await response.json();
                   addMessage('assistant', data.response);
               } catch (error) {
                   addMessage('error', 'Error: ' + error.message);
               }
           }

           function addMessage(role, content) {
               const messages = document.getElementById('messages');
               const div = document.createElement('div');
               div.className = role === 'user' ? 'chat chat-end' : 'chat chat-start';
               div.innerHTML = `<div class="chat-bubble">${content}</div>`;
               messages.appendChild(div);
               messages.scrollTop = messages.scrollHeight;
           }

           document.getElementById('userInput').addEventListener('keypress', (e) => {
               if (e.key === 'Enter') sendMessage();
           });
       </script>
   </body>
   </html>
   ```

4. **Run script** (`scripts/dev.sh`)
   ```bash
   #!/bin/bash
   export QUART_APP=app.main:app
   hypercorn app.main:app --reload --bind 0.0.0.0:5000
   ```

5. **Test it**
   ```bash
   chmod +x scripts/dev.sh
   ./scripts/dev.sh
   # Visit http://localhost:5000
   ```

**Deliverable:** ✅ Working chat UI where you can talk to Ollama directly. No RAG, no sessions—just basic chat.

**Testing checklist:**
- [ ] UI loads at http://localhost:5000
- [ ] Can send message and get response from LLM
- [ ] /health/ready returns 200
- [ ] No errors in browser console

---

## Phase 2: RAG Pipeline - Indexing ✅ COMPLETED

**Goal:** Ingest notes, chunk them, embed, store in FAISS. No chat integration yet—just build the index.

**Status:** ✅ Completed 2025-11-01

**Deliverables:**
- ✅ Database schema with `index_metadata` and `chunks` tables
- ✅ Markdown parser with frontmatter support (`app/rag/md_parser.py`)
- ✅ Character-based text chunker with overlap (`app/rag/chunker.py`)
- ✅ FAISS vector store with runtime dimension detection (`app/rag/store_faiss.py`)
- ✅ Complete ingest pipeline (`app/rag/ingest.py`)
- ✅ CLI reindex script (`scripts/reindex.py`)
- ✅ 3 sample markdown notes in `notes/` directory
- ✅ 11 chunks indexed successfully

**Key Implementation Details:**
- Runtime embedding dimension detection (no hardcoded dimensions)
- Character-based chunking (2400 chars, 320 overlap) avoids tokenizer dependencies
- FAISS IndexFlatL2 for exact nearest neighbor search
- SQLite for chunk metadata and search results
- Progress reporting with visual progress bar
- Error handling with partial success (continues if individual files fail)

### Tasks

1. **Database schema** (`app/db.py`)
   ```python
   import aiosqlite
   from pathlib import Path
   from app import config

   DB_PATH = config.DATA_DIR / "assistant.sqlite"

   async def init_db():
       async with aiosqlite.connect(DB_PATH) as db:
           await db.execute("""
               CREATE TABLE IF NOT EXISTS index_metadata (
                   id INTEGER PRIMARY KEY,
                   embedding_model TEXT NOT NULL,
                   embedding_dimension INTEGER NOT NULL,
                   created_at TEXT NOT NULL,
                   last_updated TEXT NOT NULL
               )
           """)
           await db.execute("""
               CREATE TABLE IF NOT EXISTS chunks (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   chunk_id TEXT UNIQUE NOT NULL,
                   note_path TEXT NOT NULL,
                   heading_path TEXT,
                   content TEXT NOT NULL,
                   char_count INTEGER NOT NULL,
                   created_at TEXT NOT NULL
               )
           """)
           await db.commit()
   ```

2. **FAISS store** (`app/rag/store_faiss.py`)
   - Implement `init_or_load_index()` from ARCHITECTURE.md (dimension detection)
   - `add_vectors()`, `search()`, `delete_by_ids()`

3. **Markdown parser** (`app/rag/md_parser.py`)
   - Parse frontmatter
   - Extract heading hierarchy
   - Chunk by character count (using langchain MarkdownTextSplitter)

4. **Ingest pipeline** (`app/rag/ingest.py`)
   - Walk `notes/` directory
   - Parse each .md file
   - Chunk with overlap
   - Embed via Ollama
   - Store in SQLite + FAISS

5. **Manual reindex script** (`scripts/reindex.py`)
   ```python
   import asyncio
   from app.rag.ingest import reindex_all_notes

   if __name__ == "__main__":
       asyncio.run(reindex_all_notes())
   ```

6. **Add sample notes** (`notes/`)
   ```bash
   mkdir -p notes
   cat > notes/welcome.md <<EOF
   # Welcome

   This is your note-taking system with AI-powered retrieval.

   ## Getting Started

   Add your markdown notes to the notes/ directory.
   EOF

   cat > notes/projects.md <<EOF
   # Projects

   ## Toronto AI Assistant

   A local-first RAG system with Quart and Ollama.
   EOF
   ```

7. **Test indexing**
   ```bash
   python scripts/reindex.py
   # Should see: Indexed X chunks from Y notes
   ```

**Deliverable:** Notes indexed in FAISS + SQLite. Can run reindex script successfully.

**Testing checklist:**
- [ ] `data/metadata.json` created with correct dimensions
- [ ] `data/vectors.index` FAISS file exists
- [ ] `data/assistant.sqlite` has chunks table populated
- [ ] Reindex script runs without errors

---

## Phase 3: RAG-Enhanced Chat ✅ COMPLETED

**Goal:** Integrate retrieval into chat. Chat now answers questions using your notes.

**Status:** ✅ Completed 2025-11-01

**Deliverables:**
- ✅ Retriever module with semantic search (`app/rag/retriever.py`)
- ✅ Updated `/api/chat` endpoint with RAG integration
- ✅ Context-enhanced system prompts
- ✅ Source citations in API responses
- ✅ UI updates to display sources (collapsible with relevance scores)
- ✅ Graceful degradation (chat works even if RAG fails)
- ✅ Tested with multiple queries - successfully retrieves and uses context

**Key Features:**
- Automatic context retrieval for every query (can be disabled with `use_rag: false`)
- Top-K retrieval (default: 8 chunks)
- Relevance scoring with L2 distance
- Source attribution in responses
- Collapsible source view in UI
- Content previews (200 chars) for each source

### Tasks

1. **Retriever** (`app/rag/retriever.py`)
   ```python
   async def retrieve_chunks(query: str, top_k: int = 8):
       # Embed query
       # Search FAISS
       # Fetch chunk metadata from SQLite
       # Return ranked chunks with sources
   ```

2. **Update chat endpoint** (`app/main.py`)
   ```python
   @app.route("/api/chat", methods=["POST"])
   async def chat():
       data = await request.get_json()
       user_message = data.get("message", "")

       # Retrieve relevant chunks
       chunks = await retrieve_chunks(user_message)

       # Build context
       context = "\n\n".join([
           f"[{c['note_path']}#{c['heading_path']}]\n{c['content']}"
           for c in chunks
       ])

       # System prompt with RAG
       system_prompt = f"""You are a helpful assistant with access to a knowledge base.

Answer the user's question using ONLY the information from these notes:

{context}

If the information is not in the notes, say so. Always cite your sources like [note_path#heading]."""

       messages = [
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": user_message}
       ]

       response = await llm_client.chat(messages)

       # Return response + sources
       return jsonify({
           "response": response["message"]["content"],
           "sources": [{"path": c["note_path"], "heading": c["heading_path"]} for c in chunks]
       })
   ```

3. **Update UI to show sources** (`web/templates/chat.html`)
   - Display citations below each response
   - Show which notes were referenced

**Deliverable:** ✅ Chat now retrieves and cites your notes. Ask questions about notes and get grounded answers.

**Testing checklist:**
- [ ] Asking about note content returns relevant info
- [ ] Citations appear in responses
- [ ] Asking about unknown topics says "not in notes"
- [ ] Sources displayed in UI

---

## Phase 4: Sessions & Memory (Day 6-7)

**Goal:** Multi-turn conversations with memory. Save/load sessions.

### Tasks

1. **Session schema** (`app/db.py`)
   ```sql
   CREATE TABLE sessions (
       id TEXT PRIMARY KEY,
       title TEXT,
       created_at TEXT NOT NULL
   );

   CREATE TABLE messages (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       session_id TEXT NOT NULL,
       role TEXT NOT NULL,
       content TEXT NOT NULL,
       created_at TEXT NOT NULL,
       FOREIGN KEY (session_id) REFERENCES sessions(id)
   );
   ```

2. **Conversation manager** (`app/memory/manager.py`)
   - `create_session()`
   - `add_message(session_id, role, content)`
   - `get_recent_messages(session_id, limit=6)`
   - Token-based windowing

3. **Update chat to use sessions**
   - POST /api/sessions (create)
   - GET /api/sessions (list)
   - DELETE /api/sessions/:id
   - GET /api/messages?session_id=...
   - Update /api/chat to maintain conversation history

4. **Update UI for sessions**
   - Sidebar with session list
   - "New chat" button
   - Load previous conversations

**Deliverable:** ✅ Multi-turn conversations saved and retrievable.

---

## Phase 5: Tool Calling (Day 8-9)

**Goal:** Add weather + web search tools. LLM can call tools when needed.

### Tasks

1. **Tool registry** (`app/tools/registry.py`)
   - Implement Tool dataclass from ARCHITECTURE.md
   - JSON parsing from LLM responses
   - Tool execution loop

2. **Weather tool** (`app/tools/weather.py`)
   - Call Open-Meteo API
   - Cache results (5 min TTL)

3. **Search tool** (`app/tools/websearch.py`)
   - DuckDuckGo HTTP wrapper
   - Apply SEARCH_MAX_RESULTS, domain filters

4. **Tool-aware prompt**
   - Update system prompt with tool descriptions
   - Implement tool call → execute → final answer loop

5. **UI indicators for tool use**
   - Show "Using weather tool..." loading state
   - Display tool results in chat

**Deliverable:** ✅ Can ask "What's the weather in Lisbon?" and get real data.

---

## Phase 6: File Watcher & Polish (Day 10-11)

**Goal:** Auto-reindex on note changes. Production-ready features.

### Tasks

1. **Watchdog integration** (`app/rag/ingest.py`)
   - Monitor notes/ for changes
   - Auto-reindex modified files
   - Handle deletions

2. **Observability**
   - Add structlog configuration
   - Health checks with dimension validation
   - /metrics endpoint (optional)

3. **Error handling**
   - Ollama connection failures
   - Empty retrieval fallback
   - Dimension mismatch detection

4. **E2E tests** (`tests/e2e/`)
   - Playwright test: send message, get response
   - Playwright test: RAG retrieval with sources
   - Playwright test: session persistence

5. **Polish**
   - Loading states
   - Error messages in UI
   - Dark mode toggle
   - Responsive design tweaks

**Deliverable:** ✅ Production-ready app with auto-indexing and robust error handling.

---

## Phase 7: Advanced Features (Optional)

- Streaming responses (SSE/WebSocket)
- Conversation summaries (long-term memory)
- Multi-user support (add auth)
- Export conversations
- .rag-ignore for sensitive notes
- Hybrid search (vector + FTS5)

---

## Quick Start After Phase 1

```bash
# Terminal 1: Run app
./scripts/dev.sh

# Terminal 2: Watch CSS (during dev)
./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --watch

# Visit http://localhost:5000 and start chatting!
```

---

## Testing Strategy by Phase

| Phase | Tests |
|-------|-------|
| 0 | `pytest tests/test_config.py` (validate Ollama models) |
| 1 | Manual UI testing + `curl` to /api/chat |
| 2 | `pytest tests/test_rag.py` (chunking, embedding, retrieval) |
| 3 | Manual UI testing with note questions |
| 4 | `pytest tests/test_sessions.py` |
| 5 | `pytest tests/test_tools.py` |
| 6 | `pytest tests/e2e/` (Playwright) |

---

## Commit Strategy (Following CLAUDE.md)

Each phase = 1-3 small PRs:

```
feat: scaffold project structure and dependencies
feat: add basic chat UI with Ollama integration
feat: implement RAG indexing pipeline
feat: integrate retrieval into chat
feat: add session management
feat: implement tool calling (weather + search)
feat: add file watcher and observability
```

---

## Estimated Timeline

- **Phase 0:** 2-3 hours
- **Phase 1:** 4-6 hours ⭐ **First working demo**
- **Phase 2:** 6-8 hours
- **Phase 3:** 4-5 hours ⭐ **RAG working**
- **Phase 4:** 5-6 hours
- **Phase 5:** 6-8 hours
- **Phase 6:** 4-6 hours

**Total:** ~35-45 hours of focused work (5-7 days)

**First testable UI:** End of Day 1 (Phase 1 complete)

---

## ✅ Completed Phases

### Phase 0: Bootstrap ✓
- Project structure created
- All dependencies installed (latest stable versions)
- Ollama models validated (gemma3:12b, mxbai-embed-large:latest)
- Tailwind CSS built (v4.1.16)
- Validation script created and passed

### Phase 1: Basic Chat UI ✓
- LLM client wrapper (`app/llm_client.py`)
- Quart web application (`app/main.py`)
- Chat interface with Alpine.js (minimal JS approach)
- Health check endpoints
- Working demo at http://localhost:5001

**Key Achievement:** Refactored from verbose vanilla JS to clean Alpine.js (70% code reduction)

---

## 🚀 Next Steps

You have a working AI chat interface! Here are your options:

### Option 1: Test the Current UI (Recommended)
1. Open http://localhost:5001 in your browser
2. Try chatting with the AI
3. Verify everything works as expected

### Option 2: Proceed to Phase 2 (RAG Pipeline)
Start building the note indexing system:
- Markdown parser
- Text chunking (character-based)
- Embedding generation
- FAISS vector store
- Reindex script

### Option 3: Make Adjustments
- Tweak the UI styling
- Adjust system prompts
- Change models
- Add features to Phase 1

### Option 4: Take a Break
- I can stop the server
- Review the code
- Plan Phase 2 approach

**Current server:** Running on port 5001 (background process)
