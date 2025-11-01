"""Main Quart application for Toronto AI Assistant."""
from quart import Quart, render_template, request, jsonify
import structlog
from pathlib import Path

from app import config
from app.llm_client import ollama_client
from app.rag.retriever import get_retriever
from app.memory import ConversationManager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
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
                    logger.info("no_relevant_context_found")

            except Exception as e:
                # Log RAG error but continue with chat (graceful degradation)
                logger.error(
                    "rag_retrieval_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Continue without RAG

        # Build system prompt
        if context:
            system_content = f"""You are a helpful AI assistant with access to a knowledge base.

Use the following context from the knowledge base to answer the user's question. If the context doesn't contain relevant information, you can still provide a helpful response based on your general knowledge, but mention that you don't have specific information in the knowledge base about that topic.

Context from knowledge base:
{context}

Instructions:
- Answer based on the context when relevant
- Be concise and friendly
- If you reference information from the context, you can mention the source
- If the context doesn't help, say so and provide your best answer"""
        else:
            system_content = "You are a helpful AI assistant. Be concise and friendly."

        # Build messages for LLM (system + history + current)
        # Note: history already includes the current user message we just added
        messages = [{"role": "system", "content": system_content}] + conversation_history

        # Call Ollama
        response = await ollama_client.chat(messages)

        assistant_message = response.get("message", {}).get("content", "")

        if not assistant_message:
            logger.error("empty_ollama_response", response=response)
            return jsonify({"error": "Empty response from LLM"}), 500

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
    - Required models are available
    """
    checks = {
        "status": "healthy",
        "ollama": False,
        "models": False,
    }

    try:
        # Check Ollama connectivity
        models = await ollama_client.list_models()
        checks["ollama"] = True

        # Check required models
        if config.CHAT_MODEL in models:
            checks["models"] = True
        else:
            checks["status"] = "unhealthy"
            checks["error"] = f"Missing chat model: {config.CHAT_MODEL}"

        status_code = 200 if checks["status"] == "healthy" else 503
        return jsonify(checks), status_code

    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        checks["status"] = "unhealthy"
        checks["error"] = str(e)
        return jsonify(checks), 503


@app.route("/health/live")
async def health_live():
    """Liveness probe - check if app is running."""
    return jsonify({"status": "alive"}), 200


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
