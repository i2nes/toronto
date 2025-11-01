"""Main Quart application for Toronto AI Assistant."""
from quart import Quart, render_template, request, jsonify
import structlog
from pathlib import Path

from app import config
from app.llm_client import ollama_client

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
    """Handle chat completion requests.

    Expects JSON body:
    {
        "message": "user message text"
    }

    Returns JSON:
    {
        "response": "assistant response text",
        "model": "model_name"
    }
    """
    try:
        # Get JSON data
        data = await request.get_json()

        if not data or "message" not in data:
            logger.error("missing_message_field", data=data)
            return jsonify({"error": "Missing 'message' in request body"}), 400

        user_message = data["message"].strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Limit message length (basic security)
        if len(user_message) > 2000:
            return jsonify({"error": "Message too long (max 2000 characters)"}), 400

        logger.info(
            "chat_request_received",
            message_length=len(user_message),
            user_message_preview=user_message[:100],
        )

        # Build messages for LLM (simple system prompt for now)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Be concise and friendly.",
            },
            {"role": "user", "content": user_message},
        ]

        # Call Ollama
        response = await ollama_client.chat(messages)

        assistant_message = response.get("message", {}).get("content", "")

        if not assistant_message:
            logger.error("empty_ollama_response", response=response)
            return jsonify({"error": "Empty response from LLM"}), 500

        logger.info(
            "chat_response_sent",
            response_length=len(assistant_message),
        )

        return jsonify({
            "response": assistant_message,
            "model": config.CHAT_MODEL,
        })

    except Exception as e:
        logger.error("chat_endpoint_error", error=str(e), error_type=type(e).__name__)
        return jsonify({
            "error": "An error occurred processing your request. Please try again."
        }), 500


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
