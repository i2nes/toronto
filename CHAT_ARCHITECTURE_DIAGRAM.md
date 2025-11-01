================================================================================
                     CHAT ARCHITECTURE DIAGRAM
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (chat.html)                              │
│                        Alpine.js Component                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐       │
│  │      SIDEBAR (Left)          │  │   MAIN CHAT AREA (Right)     │       │
│  ├──────────────────────────────┤  ├──────────────────────────────┤       │
│  │ [+ New Chat]                 │  │ Welcome Message              │       │
│  │ ───────────────────────────  │  │ (when messages.length === 0) │       │
│  │ ► Session 1 (2h ago)         │  │                              │       │
│  │ ► Session 2 (1d ago)         │  │ Or:                          │       │
│  │ ► Session 3 (Just now) [✓]   │  │ ┌──────────────────────────┐ │       │
│  │ (btn-active on current)      │  │ │ User: "Your message"     │ │       │
│  │                              │  │ ├──────────────────────────┤ │       │
│  │ Data Source:                 │  │ │ Bot: "Response text      │ │       │
│  │ fetchSessions() →            │  │ │ [Sources: 2 used]"       │ │       │
│  │ GET /api/sessions            │  │ └──────────────────────────┘ │       │
│  │                              │  │                              │       │
│  │ Actions:                     │  │ [Message Input Box]          │       │
│  │ • loadSession(id)            │  │ [Send Button]                │       │
│  │ • startNewChat()             │  │                              │       │
│  │ (deleteSession - missing UI) │  │                              │       │
│  │                              │  │                              │       │
│  └──────────────────────────────┘  └──────────────────────────────┘       │
│                                                                             │
│  Data:                                                                      │
│  • sessions = [{id, title, created_at}, ...]  (from GET /api/sessions)   │
│  • messages = [{id, role, content, sources}, ...]  (from API)            │
│  • currentSessionId = null (new chat) or UUID (loaded session)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ HTTP API
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    API ENDPOINTS (app/main.py)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POST   /api/chat                  (Lines 148-422)                        │
│    ├─ Input:  {message, session_id?, use_rag?}                           │
│    ├─ Action: Create session if needed                                    │
│    ├─ Action: Save user message                                          │
│    ├─ Action: LLM processing + save response                             │
│    ├─ Action: Auto-update title on first message                         │
│    └─ Output: {response, session_id, sources, model}                     │
│                                                                             │
│  POST   /api/sessions              (Lines 425-452)                        │
│    ├─ Input:  {title?}                                                    │
│    └─ Output: {id, title, created_at}                                    │
│                                                                             │
│  GET    /api/sessions              (Lines 455-473)                        │
│    └─ Output: {sessions: [{...}, ...]}                                   │
│                                                                             │
│  GET    /api/sessions/<id>/messages (Lines 497-525)                       │
│    └─ Output: {messages: [{id, role, content, sources}, ...]}            │
│                                                                             │
│  DELETE /api/sessions/<id>         (Lines 476-494)                        │
│    └─ Output: 204 No Content (success) or 404 (not found)                │
│                                                                             │
│  MISSING:                                                                   │
│  PATCH  /api/sessions/<id>         (Rename - not implemented)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ Method calls
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            CONVERSATION MANAGER (app/memory/manager.py)                    │
│                    ConversationManager Class                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  create_session(title?)           → db.create_session()                  │
│  add_message(session_id, role, content, sources?)                        │
│  get_session(session_id)          → db.get_session()                     │
│  get_all_messages(session_id)     → db.get_messages()                    │
│  get_recent_messages(session_id, limit=6)                                │
│  delete_session(session_id)       → db.delete_session()                  │
│  update_session_title(session_id, title)  ← AUTO CALLED on 1st message   │
│  format_conversation_history(session_id)  ← For LLM context              │
│                                                                             │
│  All methods log to structlog (structured logging)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ SQL operations
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  DATABASE LAYER (app/db.py)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  create_session(session_id, title)  │  get_session(session_id)           │
│  delete_session(session_id)         │  list_sessions(limit=50)           │
│  add_message(...)                   │  get_messages(session_id)          │
│  get_recent_messages(session_id)    │  [update_session_title - missing]  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ SQL Queries
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SQLite DATABASE                                       │
│                   (data/assistant.sqlite)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────┐         ┌──────────────────────────┐       │
│  │   TABLE: sessions        │         │   TABLE: messages        │       │
│  ├──────────────────────────┤         ├──────────────────────────┤       │
│  │ id        (TEXT PK)      │◄────┐   │ id         (INT PK)      │       │
│  │ title     (TEXT)         │     └───┤ session_id (FK TEXT)     │       │
│  │ created_at(TEXT)         │         │ role       (TEXT)        │       │
│  │                          │         │ content    (TEXT)        │       │
│  └──────────────────────────┘         │ sources_json (TEXT)      │       │
│                                        │ created_at (TEXT)        │       │
│                                        │                          │       │
│                                        │ Indexes:                 │       │
│                                        │ • idx_messages_session_id │       │
│                                        │ • idx_messages_created_at │       │
│                                        └──────────────────────────┘       │
│                                                                             │
│  Constraints:                                                               │
│  • messages.session_id → sessions.id (ON DELETE CASCADE)                  │
│  • messages ordered by created_at ASC (chronological)                     │
│  • sessions ordered by created_at DESC (most recent first)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
                         DATA FLOW EXAMPLES
================================================================================

1. NEW CHAT FLOW (Create Session)
   ─────────────────────────────────
   
   Frontend:                          Backend:
   ┌────────────────────────────┐    ┌─────────────────────────┐
   │ sendMessage()              │    │ POST /api/chat          │
   │ (no currentSessionId)       │───▶│ (no session_id in req)  │
   │                            │    │                         │
   │ POST {message, no id}      │    │ generate UUID ← uuid4() │
   │                            │    │ db.create_session(id)   │
   │                            │    │ db.add_message(user msg)│
   │                            │    │ [LLM processing]        │
   │                            │◀───│ db.add_message(resp)    │
   │ Receive {session_id}       │    │ UPDATE title = msg[:50] │
   │ Update currentSessionId    │    │ return {response, id}   │
   │ fetchSessions()            │    │                         │
   │ Show new session in sidebar│    │                         │
   └────────────────────────────┘    └─────────────────────────┘


2. LOAD SESSION FLOW (Resume Chat)
   ────────────────────────────────
   
   Frontend:                          Backend:
   ┌────────────────────────────┐    ┌─────────────────────────┐
   │ loadSession(sessionId)     │    │ GET /api/sessions/<id>/ │
   │                            │───▶│ messages                │
   │ currentSessionId = id      │    │                         │
   │ messages = []              │    │ SELECT * FROM messages  │
   │ GET /api/sessions/<id>/msg │    │ WHERE session_id = ?    │
   │                            │    │ ORDER BY created_at ASC │
   │                            │◀───│ return {messages: [...]}│
   │ messages = response.data   │    │                         │
   │ Sidebar button gets        │    │                         │
   │ btn-active class           │    │                         │
   └────────────────────────────┘    └─────────────────────────┘


3. DELETE SESSION FLOW (Cleanup)
   ──────────────────────────────
   
   Frontend:                          Backend:
   ┌────────────────────────────┐    ┌─────────────────────────┐
   │ deleteSession(id)          │    │ DELETE /api/sessions/<id>
   │                            │───▶│                         │
   │ DELETE /api/sessions/<id>  │    │ DELETE FROM sessions    │
   │                            │    │ WHERE id = ?            │
   │                            │    │                         │
   │                            │    │ (CASCADE: messages      │
   │                            │    │ auto-deleted)           │
   │                            │    │                         │
   │                            │◀───│ return 204 No Content   │
   │ Remove from sessions array │    │                         │
   │ Clear currentSessionId     │    │                         │
   │ Show welcome message       │    │                         │
   └────────────────────────────┘    └─────────────────────────┘


================================================================================
                    MISSING FEATURE: RENAME SESSION
================================================================================

Would require:

DATABASE:
  def update_session_title(session_id, new_title):
    UPDATE sessions SET title = ? WHERE id = ?

API:
  @app.route("/api/sessions/<id>", methods=["PATCH"])
  async def rename_session_endpoint(id):
    - Validate new_title (not empty, max 200 chars)
    - Call db.update_session_title()
    - Return updated session object

FRONTEND:
  async renameSession(sessionId, newTitle):
    - Validate input
    - PATCH /api/sessions/<id> {title: newTitle}
    - Update sessions array locally

UI:
  - Context menu (•••) on each session in sidebar
  - Options: "Rename", "Delete"
  - Modal/dialog for text input with confirmation

================================================================================
