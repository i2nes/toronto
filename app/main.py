"""Main Quart application for Toronto AI Assistant."""
from quart import Quart, render_template, request, jsonify
import structlog
import json
import re
from pathlib import Path
from datetime import datetime

from app import config, db
from app.llm_client import ollama_client
from app.rag.retriever import get_retriever
from app.memory import ConversationManager
from app.tools import get_registry
from app.rag.watcher import get_watcher, stop_watcher

# Configure structured logging
import logging
import sys

# Setup Python's logging to output to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.DATA_DIR / "app.log", mode="a"),
    ],
)

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()  # Human-readable format for development
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize Quart app
app = Quart(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "web" / "templates"),
    static_folder=str(Path(__file__).parent.parent / "web" / "static"),
)

# Initialize conversation manager
conversation_manager = ConversationManager(context_window_size=6)

# Initialize tool registry
tool_registry = get_registry()


@app.before_serving
async def startup():
    """Startup tasks - initialize file watcher."""
    try:
        # Start file watcher for auto-reindexing
        watcher = await get_watcher()
        logger.info("file_watcher_started", notes_dir=str(config.NOTES_DIR))
    except Exception as e:
        logger.error("file_watcher_start_failed", error=str(e), error_type=type(e).__name__)
        # Don't fail startup if watcher fails - app can still work without it


@app.after_serving
async def shutdown():
    """Shutdown tasks - stop file watcher."""
    try:
        await stop_watcher()
        logger.info("file_watcher_stopped")
    except Exception as e:
        logger.error("file_watcher_stop_failed", error=str(e))


def _parse_tool_call(text: str) -> dict | None:
    """Parse a tool call from LLM response.

    Looks for JSON in the format:
    {
      "tool": "tool_name",
      "args": {...}
    }

    Handles markdown code blocks (```json ... ```)

    Args:
        text: LLM response text

    Returns:
        Dict with tool and args if found, None otherwise
    """
    # Strip markdown code blocks if present
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        # Extract just the content inside code block
        text = code_block_match.group(1).strip()

    # Find the first '{' to start JSON extraction
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    # Extract JSON by matching braces from start
    brace_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if brace_count != 0:
        # Unmatched braces
        return None

    json_str = text[start_idx:end_idx]

    try:
        tool_call = json.loads(json_str)

        if "tool" in tool_call and "args" in tool_call:
            return tool_call

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("failed_to_parse_tool_call", error=str(e), text_preview=json_str[:100])

    return None


@app.route("/")
async def index():
    """Render the main chat interface."""
    return await render_template(
        "chat.html",
        static_version=config.STATIC_VERSION,
        chat_model=config.CHAT_MODEL,
    )


@app.route("/api/chat", methods=["POST"])
async def chat():
    """Handle chat completion requests with RAG and session management.

    Expects JSON body:
    {
        "message": "user message text",
        "session_id": "optional-session-id",  // creates new if not provided
        "use_rag": true  // optional, defaults to true
    }

    Returns JSON:
    {
        "response": "assistant response text",
        "model": "model_name",
        "session_id": "session-id",
        "sources": [...]  // if RAG was used
    }
    """
    try:
        # Get JSON data
        data = await request.get_json()

        if not data or "message" not in data:
            logger.error("missing_message_field", data=data)
            return jsonify({"error": "Missing 'message' in request body"}), 400

        user_message = data["message"].strip()
        session_id = data.get("session_id")
        use_rag = data.get("use_rag", True)  # RAG enabled by default

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Limit message length (basic security)
        if len(user_message) > 2000:
            return jsonify({"error": "Message too long (max 2000 characters)"}), 400

        # Create new session if not provided
        if not session_id:
            session_id = conversation_manager.create_session()
            logger.info("new_session_created", session_id=session_id)

        logger.info(
            "chat_request_received",
            session_id=session_id,
            message_length=len(user_message),
            use_rag=use_rag,
            user_message_preview=user_message[:100],
        )

        # Save user message to session
        conversation_manager.add_message(session_id, "user", user_message)

        # Get conversation history
        conversation_history = conversation_manager.format_conversation_history(session_id)

        # Retrieve context if RAG is enabled
        context = ""
        sources = []

        if use_rag:
            try:
                retriever = await get_retriever()
                results = await retriever.retrieve(user_message)

                if results:
                    # Format context for prompt
                    context = await retriever.retrieve_context(user_message)

                    # Format sources for response
                    sources = [
                        {
                            "source": result.source,
                            "content_preview": result.content[:200] + "..."
                            if len(result.content) > 200
                            else result.content,
                            "relevance": round(result.relevance_score, 3),
                        }
                        for result in results
                    ]

                    logger.info(
                        "rag_retrieval_completed",
                        num_sources=len(sources),
                        context_length=len(context),
                    )
                else:
                    # Empty retrieval - gracefully fallback to chat without RAG
                    logger.info(
                        "no_relevant_context_found",
                        query=user_message[:100],
                        message="Proceeding with general knowledge response",
                    )

            except Exception as e:
                # Log RAG error but continue with chat (graceful degradation)
                logger.error(
                    "rag_retrieval_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    message="Continuing without RAG context",
                )
                # Continue without RAG - app remains functional

        # Build system prompt with tool information
        tools_description = tool_registry.get_tools_description()

        base_instructions = """You are a helpful AI assistant. Be concise and friendly.

AVAILABLE TOOLS:
You have access to the following tools. If a tool would be helpful to answer the user's question, respond with a JSON object in this exact format:
{
  "tool": "tool_name",
  "args": {
    "arg1": "value1",
    "arg2": "value2"
  }
}

Only respond with JSON if you need to call a tool. Otherwise, respond with natural language.

"""

        if context:
            system_content = base_instructions + f"""
KNOWLEDGE BASE CONTEXT:
{context}

INSTRUCTIONS:
- First check if the knowledge base context contains relevant information
- If the question requires real-time data (weather, web search), use the appropriate tool
- Answer based on the context when relevant
- If you reference information from the context, you can mention the source
- If the context doesn't help, provide your best answer or suggest using a tool

TOOLS:
{tools_description}
"""
        else:
            system_content = base_instructions + f"""
TOOLS:
{tools_description}

INSTRUCTIONS:
- Answer questions directly when you have the knowledge
- Use tools when you need real-time data (weather, current events, web searches)
- Be helpful and concise
"""

        # Build messages for LLM (system + history + current)
        # Note: history already includes the current user message we just added
        messages = [{"role": "system", "content": system_content}] + conversation_history

        # Tool execution loop
        tool_calls = []
        max_tool_iterations = 3  # Prevent infinite loops

        for iteration in range(max_tool_iterations):
            # Call Ollama with error handling
            try:
                response = await ollama_client.chat(messages)
                assistant_message = response.get("message", {}).get("content", "")

                if not assistant_message:
                    logger.error("empty_ollama_response", response=response)
                    return jsonify({"error": "Empty response from LLM"}), 500

            except Exception as ollama_error:
                import httpx
                # Handle Ollama connection failures specifically
                if isinstance(ollama_error, httpx.ConnectError):
                    logger.error(
                        "ollama_connection_failed",
                        error=str(ollama_error),
                        base_url=config.OLLAMA_BASE_URL,
                    )
                    return jsonify({
                        "error": "Unable to connect to AI service. Please check that Ollama is running."
                    }), 503
                elif isinstance(ollama_error, httpx.TimeoutException):
                    logger.error("ollama_timeout", error=str(ollama_error), model=config.CHAT_MODEL)
                    return jsonify({
                        "error": f"AI service request timed out after 5 minutes. The model '{config.CHAT_MODEL}' may be too large or slow. Try using a smaller/faster model."
                    }), 504
                else:
                    logger.error(
                        "ollama_request_failed",
                        error=str(ollama_error),
                        error_type=type(ollama_error).__name__,
                    )
                    return jsonify({
                        "error": "AI service error. Please try again."
                    }), 500

            # Check if response contains a tool call
            tool_call = _parse_tool_call(assistant_message)

            if not tool_call:
                # No tool call, we have the final answer
                break

            # Execute the tool
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})

            logger.info("tool_call_detected", tool=tool_name, args=tool_args)

            tool_result = await tool_registry.execute_tool(tool_name, tool_args)
            tool_calls.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result.data if tool_result.success else None,
                "error": tool_result.error,
            })

            if not tool_result.success:
                # Tool failed, return error to LLM
                messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                messages.append({
                    "role": "user",
                    "content": f"Tool execution failed: {tool_result.error}. Please respond without using tools."
                })
                continue

            # Add tool result to conversation for LLM to formulate final answer
            messages.append({
                "role": "assistant",
                "content": assistant_message
            })
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' returned: {json.dumps(tool_result.data)}. Please provide a natural language answer to the user based on this data."
            })

        # If we exhausted iterations, assistant_message is the last response

        # Save assistant message to session
        conversation_manager.add_message(session_id, "assistant", assistant_message, sources)

        # Update session title with first message if this is the first exchange
        messages_in_session = conversation_manager.get_all_messages(session_id)
        if len(messages_in_session) == 2:  # First user message + first assistant response
            conversation_manager.update_session_title(session_id, user_message)

        logger.info(
            "chat_response_sent",
            session_id=session_id,
            response_length=len(assistant_message),
            used_rag=bool(context),
        )

        # Build response
        response_data = {
            "response": assistant_message,
            "model": config.CHAT_MODEL,
            "session_id": session_id,
        }

        if sources:
            response_data["sources"] = sources

        if tool_calls:
            response_data["tool_calls"] = tool_calls

        return jsonify(response_data)

    except Exception as e:
        logger.error("chat_endpoint_error", error=str(e), error_type=type(e).__name__)
        return jsonify({
            "error": "An error occurred processing your request. Please try again."
        }), 500


@app.route("/api/sessions", methods=["POST"])
async def create_session():
    """Create a new chat session.

    Expects JSON body:
    {
        "title": "optional title"
    }

    Returns JSON:
    {
        "session_id": "uuid",
        "title": "title",
        "created_at": "timestamp"
    }
    """
    try:
        data = await request.get_json() or {}
        title = data.get("title")

        session_id = conversation_manager.create_session(title)
        session = conversation_manager.get_session(session_id)

        return jsonify(session), 201

    except Exception as e:
        logger.error("session_create_error", error=str(e))
        return jsonify({"error": "Failed to create session"}), 500


@app.route("/api/sessions", methods=["GET"])
async def list_sessions():
    """List all sessions.

    Returns JSON:
    {
        "sessions": [
            {"id": "uuid", "title": "title", "created_at": "timestamp"},
            ...
        ]
    }
    """
    try:
        sessions = conversation_manager.list_sessions()
        return jsonify({"sessions": sessions})

    except Exception as e:
        logger.error("sessions_list_error", error=str(e))
        return jsonify({"error": "Failed to list sessions"}), 500


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
async def delete_session_endpoint(session_id: str):
    """Delete a session and all its messages.

    Returns:
        204 No Content if successful
        404 Not Found if session doesn't exist
    """
    try:
        deleted = conversation_manager.delete_session(session_id)

        if deleted:
            return "", 204
        else:
            return jsonify({"error": "Session not found"}), 404

    except Exception as e:
        logger.error("session_delete_error", error=str(e), session_id=session_id)
        return jsonify({"error": "Failed to delete session"}), 500


@app.route("/api/sessions/<session_id>", methods=["PATCH"])
async def rename_session_endpoint(session_id: str):
    """Rename a session.

    Expects JSON body:
    {
        "title": "new title"
    }

    Returns:
        200 OK with updated session data if successful
        400 Bad Request if title is missing or invalid
        404 Not Found if session doesn't exist
    """
    try:
        data = await request.get_json()

        if not data or "title" not in data:
            return jsonify({"error": "Missing 'title' in request body"}), 400

        new_title = data["title"].strip()

        if not new_title:
            return jsonify({"error": "Title cannot be empty"}), 400

        # Limit title length
        if len(new_title) > 100:
            return jsonify({"error": "Title too long (max 100 characters)"}), 400

        # Rename the session
        updated = conversation_manager.rename_session(session_id, new_title)

        if updated:
            session = conversation_manager.get_session(session_id)
            return jsonify(session), 200
        else:
            return jsonify({"error": "Session not found"}), 404

    except Exception as e:
        logger.error("session_rename_error", error=str(e), session_id=session_id)
        return jsonify({"error": "Failed to rename session"}), 500


@app.route("/api/sessions/<session_id>/messages", methods=["GET"])
async def get_session_messages(session_id: str):
    """Get all messages for a session.

    Returns JSON:
    {
        "messages": [
            {
                "id": 1,
                "role": "user",
                "content": "...",
                "created_at": "timestamp"
            },
            ...
        ]
    }
    """
    try:
        # Verify session exists
        session = conversation_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        messages = conversation_manager.get_all_messages(session_id)
        return jsonify({"messages": messages})

    except Exception as e:
        logger.error("messages_get_error", error=str(e), session_id=session_id)
        return jsonify({"error": "Failed to get messages"}), 500


@app.route("/health/ready")
async def health_ready():
    """Readiness probe - check if app can serve requests.

    Checks:
    - Ollama service is reachable
    - Required models are available (chat + embedding)
    - RAG index exists and dimension matches
    - Database is accessible
    """
    checks = {
        "status": "healthy",
        "ollama": False,
        "chat_model": False,
        "embedding_model": False,
        "rag_index": False,
        "database": False,
        "dimension_validation": None,
    }

    try:
        # Check Ollama connectivity and models
        models = await ollama_client.list_models()
        checks["ollama"] = True

        # Check chat model
        if config.CHAT_MODEL in models:
            checks["chat_model"] = True
        else:
            checks["status"] = "degraded"
            checks["warnings"] = checks.get("warnings", []) + [
                f"Chat model missing: {config.CHAT_MODEL}"
            ]

        # Check embedding model
        if config.EMBEDDING_MODEL in models:
            checks["embedding_model"] = True
        else:
            checks["status"] = "degraded"
            checks["warnings"] = checks.get("warnings", []) + [
                f"Embedding model missing: {config.EMBEDDING_MODEL}"
            ]

        # Check RAG index and dimension validation
        try:
            from app.rag.store_faiss import FAISSVectorStore
            from app import db

            vector_store = FAISSVectorStore()

            # Check if index exists
            if vector_store.index_path.exists() and vector_store.metadata_path.exists():
                checks["rag_index"] = True

                # Load and validate dimensions
                try:
                    await vector_store.load_index()
                    checks["dimension_validation"] = {
                        "status": "valid",
                        "dimension": vector_store.dimension,
                        "model": vector_store.embedding_model,
                        "vector_count": vector_store.index.ntotal if vector_store.index else 0,
                    }
                except ValueError as e:
                    # Dimension mismatch
                    checks["dimension_validation"] = {
                        "status": "mismatch",
                        "error": str(e),
                    }
                    checks["status"] = "unhealthy"
                    checks["error"] = "RAG dimension mismatch - reindex required"
            else:
                checks["rag_index"] = False
                checks["dimension_validation"] = {"status": "no_index"}
                checks["warnings"] = checks.get("warnings", []) + [
                    "No RAG index found - run reindex"
                ]

        except Exception as e:
            logger.error("rag_health_check_failed", error=str(e), error_type=type(e).__name__)
            checks["rag_index"] = False
            checks["dimension_validation"] = {"status": "error", "error": str(e)}

        # Check database connectivity
        try:
            from app import db
            chunk_count = db.get_chunk_count()
            checks["database"] = True
            checks["database_info"] = {"chunk_count": chunk_count}
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            checks["database"] = False
            checks["status"] = "unhealthy"
            checks["error"] = f"Database error: {str(e)}"

        # Determine final status code
        if checks["status"] == "healthy":
            status_code = 200
        elif checks["status"] == "degraded":
            status_code = 200  # Still operational but with warnings
        else:
            status_code = 503

        return jsonify(checks), status_code

    except Exception as e:
        logger.error("health_check_failed", error=str(e), error_type=type(e).__name__)
        checks["status"] = "unhealthy"
        checks["error"] = str(e)
        return jsonify(checks), 503


@app.route("/health/live")
async def health_live():
    """Liveness probe - check if app is running."""
    return jsonify({"status": "alive"}), 200


@app.route("/metrics")
async def metrics():
    """Metrics endpoint for observability.

    Returns:
        JSON with application metrics including sessions, messages,
        RAG index stats, and database statistics
    """
    try:
        from app import db
        from app.rag.store_faiss import FAISSVectorStore

        metrics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "toronto-ai-assistant",
            "version": config.STATIC_VERSION,
        }

        # Session metrics
        try:
            all_sessions = db.list_sessions(limit=1000)  # Get all sessions
            metrics_data["sessions"] = {
                "total": len(all_sessions),
                "recent_count": len(db.list_sessions(limit=10)),
            }
        except Exception as e:
            logger.error("sessions_metrics_failed", error=str(e))
            metrics_data["sessions"] = {"error": str(e)}

        # Message metrics
        try:
            # Get total message count across all sessions
            from app.db import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            conn.close()

            metrics_data["messages"] = {
                "total": total_messages,
            }
        except Exception as e:
            logger.error("messages_metrics_failed", error=str(e))
            metrics_data["messages"] = {"error": str(e)}

        # RAG index metrics
        try:
            vector_store = FAISSVectorStore()
            if vector_store.index_path.exists():
                await vector_store.load_index()
                stats = vector_store.get_stats()
                metrics_data["rag_index"] = {
                    "vector_count": stats.get("vector_count", 0),
                    "dimension": stats.get("dimension"),
                    "embedding_model": stats.get("embedding_model"),
                    "index_exists": stats.get("index_exists_on_disk", False),
                }
            else:
                metrics_data["rag_index"] = {
                    "status": "not_initialized",
                    "vector_count": 0,
                }
        except Exception as e:
            logger.error("rag_metrics_failed", error=str(e))
            metrics_data["rag_index"] = {"error": str(e)}

        # Database metrics
        try:
            chunk_count = db.get_chunk_count()
            index_metadata = db.get_latest_index_metadata()

            metrics_data["database"] = {
                "chunk_count": chunk_count,
                "last_indexed_at": index_metadata.get("indexed_at") if index_metadata else None,
                "total_notes": index_metadata.get("total_notes") if index_metadata else None,
            }
        except Exception as e:
            logger.error("database_metrics_failed", error=str(e))
            metrics_data["database"] = {"error": str(e)}

        # Model configuration
        metrics_data["models"] = {
            "chat_model": config.CHAT_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        }

        # Configuration
        metrics_data["config"] = {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
            "max_context_tokens": config.MAX_CONTEXT_TOKENS,
        }

        return jsonify(metrics_data), 200

    except Exception as e:
        logger.error("metrics_endpoint_error", error=str(e), error_type=type(e).__name__)
        return jsonify({
            "error": "Failed to generate metrics",
            "message": str(e)
        }), 500


@app.route("/api/models")
async def get_models():
    """Get list of available Ollama models.

    Returns:
        JSON with available models and current selection
    """
    try:
        import httpx

        # Fetch available models from Ollama
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()

            # Extract model names
            models = [model["name"] for model in data.get("models", [])]

            return jsonify({
                "models": models,
                "current": config.CHAT_MODEL
            }), 200

    except Exception as e:
        logger.error("get_models_failed", error=str(e))
        return jsonify({
            "error": "Failed to fetch models",
            "message": str(e)
        }), 500


@app.route("/api/models/select", methods=["POST"])
async def select_model():
    """Update the selected chat model.

    Expects JSON body:
    {
        "model": "model_name"
    }
    """
    try:
        data = await request.get_json()

        if not data or "model" not in data:
            return jsonify({"error": "Missing 'model' in request body"}), 400

        model_name = data["model"].strip()

        if not model_name:
            return jsonify({"error": "Model name cannot be empty"}), 400

        # Update the config
        config.CHAT_MODEL = model_name

        logger.info("chat_model_updated", model=model_name)

        return jsonify({
            "success": True,
            "model": model_name
        }), 200

    except Exception as e:
        logger.error("select_model_failed", error=str(e))
        return jsonify({
            "error": "Failed to update model",
            "message": str(e)
        }), 500


@app.route("/notes")
async def notes_page():
    """Render the notes management interface."""
    return await render_template(
        "notes.html",
        static_version=config.STATIC_VERSION,
    )


@app.route("/api/notes", methods=["GET"])
async def list_notes():
    """List all markdown notes in the notes directory.

    Returns JSON:
    {
        "notes": [
            {
                "path": "relative/path.md",
                "name": "filename.md",
                "size": 1234,
                "modified_at": "2024-01-01T00:00:00"
            },
            ...
        ]
    }
    """
    try:
        notes = []
        notes_dir = config.NOTES_DIR

        if not notes_dir.exists():
            return jsonify({"notes": []}), 200

        # Find all markdown files
        for md_file in sorted(notes_dir.rglob("*.md")):
            try:
                relative_path = md_file.relative_to(notes_dir)
                stat = md_file.stat()

                notes.append({
                    "path": str(relative_path),
                    "name": md_file.name,
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except Exception as e:
                logger.warning("failed_to_stat_file", file=str(md_file), error=str(e))
                continue

        return jsonify({"notes": notes}), 200

    except Exception as e:
        logger.error("list_notes_failed", error=str(e), error_type=type(e).__name__)
        return jsonify({"error": "Failed to list notes"}), 500


@app.route("/api/notes/<path:note_path>", methods=["GET"])
async def get_note(note_path: str):
    """Get a specific note's content.

    Returns JSON:
    {
        "path": "path.md",
        "content": "markdown content",
        "size": 1234,
        "modified_at": "2024-01-01T00:00:00"
    }
    """
    try:
        # Validate and construct path
        notes_dir = config.NOTES_DIR
        file_path = (notes_dir / note_path).resolve()

        # Security: ensure path is within notes directory
        if not str(file_path).startswith(str(notes_dir)):
            return jsonify({"error": "Invalid path"}), 400

        if not file_path.exists():
            return jsonify({"error": "Note not found"}), 404

        if not file_path.suffix == ".md":
            return jsonify({"error": "Not a markdown file"}), 400

        # Read content
        content = file_path.read_text(encoding="utf-8")
        stat = file_path.stat()

        return jsonify({
            "path": note_path,
            "content": content,
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }), 200

    except Exception as e:
        logger.error("get_note_failed", path=note_path, error=str(e))
        return jsonify({"error": "Failed to read note"}), 500


@app.route("/api/notes/<path:note_path>/metadata", methods=["GET"])
async def get_note_metadata(note_path: str):
    """Get metadata about a note's indexing status.

    Returns JSON:
    {
        "path": "path.md",
        "indexed": true,
        "chunk_count": 5,
        "last_indexed_at": "2024-01-01T00:00:00"
    }
    """
    try:
        # Check if note has chunks in database
        chunks = db.get_chunk_ids_for_file(note_path)
        has_chunks = len(chunks) > 0

        # Get latest index metadata
        index_meta = db.get_latest_index_metadata()

        return jsonify({
            "path": note_path,
            "indexed": has_chunks,
            "chunk_count": len(chunks),
            "last_indexed_at": index_meta.get("indexed_at") if index_meta else None,
        }), 200

    except Exception as e:
        logger.error("get_note_metadata_failed", path=note_path, error=str(e))
        return jsonify({"error": "Failed to get metadata"}), 500


@app.route("/api/notes", methods=["POST"])
async def create_note():
    """Create a new note.

    Expects JSON body:
    {
        "path": "folder/filename.md",
        "content": "markdown content"
    }

    Returns JSON:
    {
        "path": "folder/filename.md",
        "created": true
    }
    """
    try:
        data = await request.get_json()

        if not data or "path" not in data or "content" not in data:
            return jsonify({"error": "Missing 'path' or 'content'"}), 400

        note_path = data["path"]
        content = data["content"]

        # Validate path
        if not note_path.endswith(".md"):
            return jsonify({"error": "Note must have .md extension"}), 400

        # Construct full path
        notes_dir = config.NOTES_DIR
        file_path = (notes_dir / note_path).resolve()

        # Security: ensure path is within notes directory
        if not str(file_path).startswith(str(notes_dir)):
            return jsonify({"error": "Invalid path"}), 400

        # Check if file already exists
        if file_path.exists():
            return jsonify({"error": "Note already exists"}), 409

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        file_path.write_text(content, encoding="utf-8")

        logger.info("note_created", path=note_path)

        return jsonify({
            "path": note_path,
            "created": True,
        }), 201

    except Exception as e:
        logger.error("create_note_failed", error=str(e))
        return jsonify({"error": "Failed to create note"}), 500


@app.route("/api/notes/<path:note_path>", methods=["PUT"])
async def update_note(note_path: str):
    """Update an existing note.

    Expects JSON body:
    {
        "content": "updated markdown content"
    }

    Returns JSON:
    {
        "path": "path.md",
        "updated": true
    }
    """
    try:
        data = await request.get_json()

        if not data or "content" not in data:
            return jsonify({"error": "Missing 'content'"}), 400

        content = data["content"]

        # Validate and construct path
        notes_dir = config.NOTES_DIR
        file_path = (notes_dir / note_path).resolve()

        # Security: ensure path is within notes directory
        if not str(file_path).startswith(str(notes_dir)):
            return jsonify({"error": "Invalid path"}), 400

        if not file_path.exists():
            return jsonify({"error": "Note not found"}), 404

        if not file_path.suffix == ".md":
            return jsonify({"error": "Not a markdown file"}), 400

        # Write updated content
        file_path.write_text(content, encoding="utf-8")

        logger.info("note_updated", path=note_path)

        return jsonify({
            "path": note_path,
            "updated": True,
        }), 200

    except Exception as e:
        logger.error("update_note_failed", path=note_path, error=str(e))
        return jsonify({"error": "Failed to update note"}), 500


@app.route("/api/notes/<path:note_path>", methods=["DELETE"])
async def delete_note(note_path: str):
    """Delete a note.

    Returns:
        204 No Content if successful
        404 Not Found if note doesn't exist
    """
    try:
        # Validate and construct path
        notes_dir = config.NOTES_DIR
        file_path = (notes_dir / note_path).resolve()

        # Security: ensure path is within notes directory
        if not str(file_path).startswith(str(notes_dir)):
            return jsonify({"error": "Invalid path"}), 400

        if not file_path.exists():
            return jsonify({"error": "Note not found"}), 404

        if not file_path.suffix == ".md":
            return jsonify({"error": "Not a markdown file"}), 400

        # Delete file
        file_path.unlink()

        logger.info("note_deleted", path=note_path)

        return "", 204

    except Exception as e:
        logger.error("delete_note_failed", path=note_path, error=str(e))
        return jsonify({"error": "Failed to delete note"}), 500


@app.route("/todos")
async def todos_page():
    """Render the todos management interface."""
    return await render_template(
        "todos.html",
        static_version=config.STATIC_VERSION,
    )


@app.route("/api/todos", methods=["GET"])
async def list_todos_endpoint():
    """List all todos, optionally filtered by completion status.

    Query params:
        completed: 'true', 'false', or omitted for all

    Returns JSON:
    {
        "todos": [
            {
                "id": 1,
                "title": "...",
                "description": "...",
                "completed": false,
                "created_at": "...",
                "updated_at": "..."
            },
            ...
        ]
    }
    """
    try:
        # Get optional filter
        completed_filter = request.args.get("completed")
        completed = None

        if completed_filter == "true":
            completed = True
        elif completed_filter == "false":
            completed = False

        todos = db.list_todos(completed=completed)

        return jsonify({"todos": todos}), 200

    except Exception as e:
        logger.error("list_todos_failed", error=str(e))
        return jsonify({"error": "Failed to list todos"}), 500


@app.route("/api/todos", methods=["POST"])
async def create_todo_endpoint():
    """Create a new todo.

    Expects JSON body:
    {
        "title": "...",
        "description": "..." (optional)
    }

    Returns JSON:
    {
        "id": 1,
        "title": "...",
        "description": "...",
        "completed": false,
        "created_at": "...",
        "updated_at": "..."
    }
    """
    try:
        data = await request.get_json()

        if not data or "title" not in data:
            return jsonify({"error": "Missing 'title'"}), 400

        title = data["title"].strip()
        if not title:
            return jsonify({"error": "Title cannot be empty"}), 400

        description = data.get("description")
        if description:
            description = description.strip() or None
        else:
            description = None

        # Create todo
        todo_id = db.create_todo(title=title, description=description)

        # Return the created todo
        todo = db.get_todo(todo_id)

        return jsonify(todo), 201

    except Exception as e:
        logger.error("create_todo_failed", error=str(e))
        return jsonify({"error": "Failed to create todo"}), 500


@app.route("/api/todos/<int:todo_id>", methods=["GET"])
async def get_todo_endpoint(todo_id: int):
    """Get a specific todo.

    Returns JSON:
    {
        "id": 1,
        "title": "...",
        "description": "...",
        "completed": false,
        "created_at": "...",
        "updated_at": "..."
    }
    """
    try:
        todo = db.get_todo(todo_id)

        if not todo:
            return jsonify({"error": "Todo not found"}), 404

        return jsonify(todo), 200

    except Exception as e:
        logger.error("get_todo_failed", error=str(e), todo_id=todo_id)
        return jsonify({"error": "Failed to get todo"}), 500


@app.route("/api/todos/<int:todo_id>", methods=["PATCH"])
async def update_todo_endpoint(todo_id: int):
    """Update a todo.

    Expects JSON body (all fields optional):
    {
        "title": "...",
        "description": "...",
        "completed": true/false
    }

    Returns JSON:
    {
        "id": 1,
        "title": "...",
        "description": "...",
        "completed": false,
        "created_at": "...",
        "updated_at": "..."
    }
    """
    try:
        data = await request.get_json()

        if not data:
            return jsonify({"error": "Missing request body"}), 400

        # Extract fields
        title = data.get("title")
        description = data.get("description")
        completed = data.get("completed")

        # Validate title if provided
        if title is not None:
            title = title.strip()
            if not title:
                return jsonify({"error": "Title cannot be empty"}), 400

        # Update todo
        updated = db.update_todo(
            todo_id=todo_id,
            title=title,
            description=description,
            completed=completed,
        )

        if not updated:
            return jsonify({"error": "Todo not found"}), 404

        # Return updated todo
        todo = db.get_todo(todo_id)

        return jsonify(todo), 200

    except Exception as e:
        logger.error("update_todo_failed", error=str(e), todo_id=todo_id)
        return jsonify({"error": "Failed to update todo"}), 500


@app.route("/api/todos/<int:todo_id>", methods=["DELETE"])
async def delete_todo_endpoint(todo_id: int):
    """Delete a todo.

    Returns:
        204 No Content if successful
        404 Not Found if todo doesn't exist
    """
    try:
        deleted = db.delete_todo(todo_id)

        if not deleted:
            return jsonify({"error": "Todo not found"}), 404

        return "", 204

    except Exception as e:
        logger.error("delete_todo_failed", error=str(e), todo_id=todo_id)
        return jsonify({"error": "Failed to delete todo"}), 500


@app.errorhandler(404)
async def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
async def internal_error(error):
    """Handle 500 errors."""
    logger.error("internal_server_error", error=str(error))
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # For development - use hypercorn via scripts/dev.sh in production
    app.run(host="0.0.0.0", port=5000, debug=True)
