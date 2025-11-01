# Chat History Implementation Analysis

## Overview
The Toronto AI Assistant uses SQLite for persistent chat storage with a clean separation between database layer, conversation manager, API endpoints, and frontend UI.

---

## 1. DATABASE SCHEMA & MODELS

### File: `/Users/mini/github/toronto/app/db.py`

#### Sessions Table (Lines 89-96)
```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,           -- UUID string
    title TEXT,                    -- Session title (auto-generated from first message)
    created_at TEXT NOT NULL       -- ISO 8601 timestamp
)
```

#### Messages Table (Lines 98-109)
```sql
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,      -- Foreign key to sessions.id
    role TEXT NOT NULL,            -- 'user' or 'assistant'
    content TEXT NOT NULL,         -- The message content
    sources_json TEXT,             -- JSON array of RAG sources used
    created_at TEXT NOT NULL,      -- ISO 8601 timestamp
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
)
```

#### Indexes
- `idx_messages_session_id` (Line 113-115): Fast lookups by session
- `idx_messages_created_at` (Line 118-121): Ordering by timestamp

#### Key Database Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `create_session()` | Lines 442-474 | Create new chat session |
| `get_session()` | Lines 477-503 | Retrieve session by ID |
| `list_sessions()` | Lines 506-533 | List all sessions (most recent first) |
| `delete_session()` | Lines 536-562 | Delete session + cascade delete messages |
| `add_message()` | Lines 565-607 | Add message to session |
| `get_messages()` | Lines 610-645 | Get all messages for a session |
| `get_recent_messages()` | Lines 648-685 | Get last N messages (for context window) |

---

## 2. CONVERSATION MANAGER (Session Orchestration)

### File: `/Users/mini/github/toronto/app/memory/manager.py`

The `ConversationManager` class wraps database operations with logging and higher-level logic.

#### Key Methods

| Method | Lines | Purpose |
|--------|-------|---------|
| `create_session()` | 26-38 | Generate UUID, create session, log |
| `add_message()` | 40-65 | Add message with sources, log |
| `get_recent_messages()` | 67-86 | Get last N messages (default: 6) |
| `get_all_messages()` | 88-103 | Get entire conversation |
| `format_conversation_history()` | 105-129 | Format for LLM (role + content only) |
| `get_session()` | 131-140 | Get session metadata |
| `list_sessions()` | 142-151 | List all sessions |
| `delete_session()` | 153-165 | Delete session with logging |
| `update_session_title()` | 167-195 | Auto-generate title from first message |

**Note on Title Updates**: Titles are truncated at 50 characters. If a word is cut off, it backtracks to the last space.

---

## 3. API ENDPOINTS

### File: `/Users/mini/github/toronto/app/main.py`

#### POST `/api/chat` (Lines 148-422)
- **Request**: `{"message": "...", "session_id": "...", "use_rag": true}`
- **Response**: `{"response": "...", "session_id": "...", "model": "...", "sources": [...]}`
- **Behavior**:
  - Creates session if not provided (Line 187-189)
  - Saves user message (Line 200)
  - Retrieves + saves assistant response (Line 389)
  - Auto-updates title on first exchange (Lines 391-394)

#### POST `/api/sessions` (Lines 425-452)
- **Create new session** with optional title
- **Response**: `{"id": "...", "title": "...", "created_at": "..."}`

#### GET `/api/sessions` (Lines 455-473)
- **List all sessions** ordered by most recent
- **Response**: `{"sessions": [...]}`

#### DELETE `/api/sessions/<session_id>` (Lines 476-494)
- **Delete session** (cascades to messages)
- **Returns**: 204 No Content (success) or 404 Not Found

#### GET `/api/sessions/<session_id>/messages` (Lines 497-525)
- **Get all messages** for a session
- **Response**: `{"messages": [...]}`

---

## 4. FRONTEND UI

### File: `/Users/mini/github/toronto/web/templates/chat.html`

#### Sidebar (Lines 79-113)
- **Sessions List**: Displays all sessions with title + relative timestamp
- **New Chat Button**: Clears messages, creates new session
- **Scroll Container**: overflow-y-auto for many sessions

#### Main Chat Area (Lines 189-285)
- **Messages**: Role-based styling (user vs assistant vs error)
- **Markdown Rendering**: Using marked.js + DOMPurify sanitization
- **Sources Collapsible**: Shows RAG sources with relevance scores
- **Tool Calls**: Displays executed tools

#### JavaScript Alpine.js App (Lines 350-562)

**Data Structure**:
```javascript
{
    messages: [],           // Array of {id, role, content, sources, tool_calls}
    sessions: [],          // Array of {id, title, created_at}
    currentSessionId: null, // Active session (null = new chat)
    userInput: '',
    isProcessing: false,
    availableModels: [],
    currentModel: '{{ chat_model }}'
}
```

**Key Methods**:

| Method | Lines | Purpose |
|--------|-------|---------|
| `init()` | 362-366 | Load sessions, models, focus input |
| `fetchSessions()` | 440-448 | GET /api/sessions |
| `startNewChat()` | 450-454 | Clear messages, set currentSessionId = null |
| `loadSession()` | 456-475 | Fetch messages for selected session |
| `sendMessage()` | 492-550 | POST /api/chat, update UI |
| `renderMarkdown()` | 368-395 | Parse + sanitize markdown |
| `formatDate()` | 477-490 | Relative timestamps (e.g., "2m ago") |

**Session UI Binding** (Lines 94-107):
```html
<template x-for="session in sessions" :key="session.id">
    <button @click="loadSession(session.id)"
            :class="{'btn-active': currentSessionId === session.id}">
        <div x-text="session.title"></div>
        <div x-text="formatDate(session.created_at)"></div>
    </button>
</template>
```

---

## 5. CHAT OPERATIONS SUMMARY

### Current Operations Implemented

1. **Create Chat Session**
   - Auto-triggered on first message (POST /api/chat)
   - Or explicit: POST /api/sessions
   - Title auto-generated from first user message

2. **Send Message**
   - POST /api/chat with message + optional session_id
   - Saves both user + assistant messages
   - Retrieves RAG sources if enabled

3. **List Sessions**
   - GET /api/sessions
   - Ordered by created_at DESC
   - Sidebar displays with timestamps

4. **Load Session**
   - GET /api/sessions/<id>/messages
   - Displays all messages in session
   - Sidebar highlights active session (btn-active)

5. **Delete Session**
   - DELETE /api/sessions/<id>
   - Cascades delete to all messages
   - Removes from sidebar (after refresh)

### Currently NOT Implemented

1. **Rename Session**: No PUT/PATCH endpoint to update title
2. **Update Session Metadata**: No endpoint to change title after creation
3. **Session Sharing/Export**: No export functionality
4. **Session Search/Filter**: Sessions list only ordered by date

---

## 6. WHERE TO ADD RENAME/DELETE FUNCTIONALITY

### For Rename Feature

**Database Layer** (`/Users/mini/github/toronto/app/db.py`):
- Add function after `delete_session()` (line 562):
```python
def update_session_title(session_id: str, new_title: str) -> bool:
    """Update session title."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (new_title, session_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        conn.rollback()
        logger.error("session_title_update_failed", error=str(e))
        raise
    finally:
        conn.close()
```

**ConversationManager** (`/Users/mini/github/toronto/app/memory/manager.py`):
- Add method after `delete_session()` (line 165):
```python
def rename_session(self, session_id: str, new_title: str) -> bool:
    """Rename a session."""
    result = db.update_session_title(session_id, new_title)
    if result:
        logger.info("session_renamed", session_id=session_id, new_title=new_title)
    return result
```

**API Endpoint** (`/Users/mini/github/toronto/app/main.py`):
- Add after `delete_session_endpoint()` (line 494):
```python
@app.route("/api/sessions/<session_id>", methods=["PATCH"])
async def rename_session_endpoint(session_id: str):
    """Rename a session."""
    try:
        data = await request.get_json() or {}
        new_title = data.get("title", "").strip()
        
        if not new_title:
            return jsonify({"error": "Title cannot be empty"}), 400
        
        if len(new_title) > 200:
            return jsonify({"error": "Title too long"}), 400
        
        success = conversation_manager.rename_session(session_id, new_title)
        if success:
            session = conversation_manager.get_session(session_id)
            return jsonify(session), 200
        else:
            return jsonify({"error": "Session not found"}), 404
    
    except Exception as e:
        logger.error("session_rename_error", error=str(e))
        return jsonify({"error": "Failed to rename session"}), 500
```

**Frontend** (`/Users/mini/github/toronto/web/templates/chat.html`):
- Add method to Alpine.js chatApp() component (after `loadSession()`, line 475):
```javascript
async renameSession(sessionId, newTitle) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle })
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to rename');
        }
        
        // Update session in list
        const idx = this.sessions.findIndex(s => s.id === sessionId);
        if (idx >= 0) {
            this.sessions[idx].title = newTitle;
        }
    } catch (err) {
        this.error = `Failed to rename: ${err.message}`;
    }
}
```

- Add UI for rename (context menu/button on session items):
```html
<div class="dropdown dropdown-end">
    <button tabindex="0" class="btn btn-ghost btn-sm btn-circle">
        <i data-lucide="more-vertical" class="w-4 h-4"></i>
    </button>
    <ul tabindex="0" class="dropdown-content menu p-2 shadow bg-base-100">
        <li><a @click="showRenameDialog(session)">Rename</a></li>
        <li><a @click="deleteSession(session.id)">Delete</a></li>
    </ul>
</div>
```

### For Delete Feature (Already Partially Implemented)

**Current Status**: Backend DELETE endpoint exists (line 476-494 in main.py)

**What's Missing**: Frontend delete button and confirmation dialog
- Add delete method + UI in chat.html
- Add confirmation before deletion
- Refresh session list after delete

---

## 7. KEY INTEGRATION POINTS

### Database to API Flow
```
db.create_session()
  → ConversationManager.create_session()
  → POST /api/chat creates session automatically
  → Frontend refreshes sessions list via GET /api/sessions
```

### Message Flow
```
User Types → sendMessage() → POST /api/chat {message, session_id}
  → conversation_manager.add_message(session_id, "user", content)
  → LLM processing
  → conversation_manager.add_message(session_id, "assistant", response)
  → Response returned + UI updates
```

### Session Persistence
```
SQLite: sessions.id (TEXT PRIMARY KEY)
↓
Messages: messages.session_id (FOREIGN KEY)
↓
API: /api/sessions/<id>/messages (cascade on delete)
↓
Frontend: currentSessionId tracks active session
↓
Sidebar: x-for loops sessions array
```

---

## 8. TEST FILES

### E2E Tests: `/Users/mini/github/toronto/tests/e2e/test_sessions.py`
- Lines 12-37: New chat creates session
- Lines 40-52: Session timestamp display
- Lines 81-114: Load previous session
- Lines 117-129: Active session highlighting
- Lines 132-148: Session persists messages
- Lines 151-165: Title based on first message
- Lines 168-187: Multiple sessions listed
- Lines 190-194: Session list scrollable

### Chat Tests: `/Users/mini/github/toronto/tests/e2e/test_basic_chat.py`
- Lines 58-77: Send message and receive response
- Lines 100-110: Character counter
- Lines 113-129: Multiple messages in sequence

---

## Summary Table

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| Create Session | ✅ Complete | db.py:442, main.py:425 | Auto or explicit |
| List Sessions | ✅ Complete | db.py:506, main.py:455 | Ordered DESC by date |
| Load Session | ✅ Complete | db.py:610, main.py:497 | Shows all messages |
| Delete Session | ✅ Complete | db.py:536, main.py:476 | Cascades to messages |
| Auto-title | ✅ Complete | manager.py:167 | From first message |
| Rename Session | ❌ Missing | N/A | Needs db + API + UI |
| Export Session | ❌ Missing | N/A | Future enhancement |
| Search Sessions | ❌ Missing | N/A | Future enhancement |
| Duplicate Session | ❌ Missing | N/A | Future enhancement |

