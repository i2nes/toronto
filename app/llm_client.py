"""Ollama LLM client wrapper with error handling."""
import httpx
from typing import List, Dict, Optional
import structlog

from app import config

logger = structlog.get_logger()


class OllamaClient:
    """Async client for interacting with Ollama API."""

    def __init__(self, base_url: str = None, timeout: float = 60.0):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (defaults to config.OLLAMA_BASE_URL)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.timeout = timeout

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> Dict:
        """Send chat completion request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to config.CHAT_MODEL)
            stream: Whether to stream the response (not implemented in Phase 1)
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            Response dict with 'message' containing 'content'

        Raises:
            httpx.HTTPError: On API errors
            httpx.ConnectError: If Ollama service is unavailable
        """
        model = model or config.CHAT_MODEL

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(
                    "ollama_chat_request",
                    model=model,
                    message_count=len(messages),
                    stream=stream,
                )

                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()

                logger.info(
                    "ollama_chat_response",
                    model=model,
                    response_length=len(data.get("message", {}).get("content", "")),
                )

                return data

        except httpx.ConnectError as e:
            logger.error("ollama_connection_error", error=str(e), base_url=self.base_url)
            raise
        except httpx.HTTPError as e:
            logger.error("ollama_http_error", error=str(e), status_code=getattr(e.response, 'status_code', None))
            raise

    async def embeddings(
        self,
        prompt: str,
        model: str = None,
    ) -> Dict:
        """Generate embeddings for a text prompt.

        Args:
            prompt: Text to embed
            model: Model to use (defaults to config.EMBEDDING_MODEL)

        Returns:
            Response dict with 'embedding' list

        Raises:
            httpx.HTTPError: On API errors
        """
        model = model or config.EMBEDDING_MODEL

        payload = {
            "model": model,
            "prompt": prompt,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug(
                    "ollama_embedding_request",
                    model=model,
                    prompt_length=len(prompt),
                )

                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()

                logger.debug(
                    "ollama_embedding_response",
                    model=model,
                    dimension=len(data.get("embedding", [])),
                )

                return data

        except httpx.HTTPError as e:
            logger.error("ollama_embedding_error", error=str(e))
            raise

    async def list_models(self) -> List[str]:
        """List all available Ollama models.

        Returns:
            List of model names

        Raises:
            httpx.HTTPError: On API errors
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as e:
            logger.error("ollama_list_models_error", error=str(e))
            raise


# Global client instance
ollama_client = OllamaClient()
