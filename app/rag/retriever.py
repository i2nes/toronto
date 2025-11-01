"""Retriever for semantic search over indexed notes.

Handles:
- Query embedding generation
- FAISS vector search
- Chunk retrieval from database
- Result ranking and formatting
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

from app import config, db
from app.llm_client import ollama_client
from app.rag.store_faiss import FAISSVectorStore

logger = structlog.get_logger()


@dataclass
class RetrievalResult:
    """A single retrieved chunk with metadata."""

    chunk_id: int
    content: str
    note_path: str
    heading_context: str
    distance: float
    metadata: Dict[str, Any]

    @property
    def source(self) -> str:
        """Get a formatted source string for display."""
        if self.heading_context:
            return f"{self.note_path} > {self.heading_context}"
        return self.note_path

    @property
    def relevance_score(self) -> float:
        """Convert L2 distance to a 0-1 relevance score.

        Lower distance = higher relevance.
        We use a simple exponential decay for interpretability.
        """
        # For L2 distance, typical values range from 0 (identical) to ~2.0 (very different)
        # We'll map this to 0-1 scale using exponential decay
        import math
        return math.exp(-self.distance / 2.0)


class Retriever:
    """Semantic retriever for RAG pipeline."""

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedding_model: str = None,
        top_k: int = None,
    ):
        """Initialize the retriever.

        Args:
            vector_store: FAISS vector store (will load default if not provided)
            embedding_model: Embedding model name (default from config)
            top_k: Number of results to retrieve (default from config)
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.top_k = top_k or config.RETRIEVAL_TOP_K

        logger.info(
            "retriever_initialized",
            embedding_model=self.embedding_model,
            top_k=self.top_k,
        )

    async def _ensure_vector_store(self) -> FAISSVectorStore:
        """Ensure vector store is loaded."""
        if self.vector_store is None:
            self.vector_store = FAISSVectorStore(embedding_model=self.embedding_model)
            await self.vector_store.init_or_load()
        return self.vector_store

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_relevance: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query text
            top_k: Number of results to return (overrides default)
            min_relevance: Minimum relevance score (0-1) to include results

        Returns:
            List of RetrievalResult objects, sorted by relevance (best first)

        Raises:
            RuntimeError: If retrieval fails
        """
        if not query or not query.strip():
            logger.warning("empty_query_provided")
            return []

        top_k = top_k or self.top_k

        logger.info("retrieval_started", query_length=len(query), top_k=top_k)

        try:
            # Ensure vector store is loaded
            store = await self._ensure_vector_store()

            # Check if index is empty
            if store.index.ntotal == 0:
                logger.warning("empty_index_no_results")
                return []

            # Generate query embedding
            response = await ollama_client.embeddings(
                model=self.embedding_model,
                prompt=query,
            )
            query_embedding = response.get("embedding", [])

            if not query_embedding:
                raise RuntimeError("Empty embedding returned for query")

            logger.debug(
                "query_embedded",
                dimension=len(query_embedding),
                model=self.embedding_model,
            )

            # Search FAISS index
            vector_ids, distances = await store.search(query_embedding, top_k=top_k)

            if not vector_ids:
                logger.info("no_results_found")
                return []

            logger.debug("vector_search_completed", results_found=len(vector_ids))

            # Retrieve chunks from database
            chunks = db.get_chunks_by_vector_ids(vector_ids)

            # Build results
            results = []
            for chunk_data in chunks:
                # Find corresponding distance
                vector_id = chunk_data["vector_id"]
                try:
                    idx = vector_ids.index(vector_id)
                    distance = distances[idx]
                except (ValueError, IndexError):
                    logger.warning(
                        "vector_id_not_found_in_search_results",
                        vector_id=vector_id,
                    )
                    continue

                result = RetrievalResult(
                    chunk_id=chunk_data["id"],
                    content=chunk_data["content"],
                    note_path=chunk_data["note_path"],
                    heading_context=chunk_data["heading_context"] or "",
                    distance=distance,
                    metadata=chunk_data.get("metadata", {}),
                )

                # Filter by minimum relevance if specified
                if result.relevance_score >= min_relevance:
                    results.append(result)

            # Sort by distance (lower = more relevant)
            results.sort(key=lambda r: r.distance)

            logger.info(
                "retrieval_completed",
                query_length=len(query),
                results_returned=len(results),
                top_distance=results[0].distance if results else None,
            )

            return results

        except Exception as e:
            logger.error(
                "retrieval_failed",
                error=str(e),
                error_type=type(e).__name__,
                query_preview=query[:100],
            )
            raise RuntimeError(f"Retrieval failed: {e}") from e

    async def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_chars: int = 4000,
    ) -> str:
        """Retrieve and format context for LLM prompt.

        Args:
            query: User query text
            top_k: Number of results to retrieve
            max_chars: Maximum total characters of context to return

        Returns:
            Formatted context string ready for LLM prompt
        """
        results = await self.retrieve(query, top_k=top_k)

        if not results:
            return ""

        # Build context string
        context_parts = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            # Format each chunk with source
            chunk_text = (
                f"[Source {i}: {result.source}]\n"
                f"{result.content.strip()}\n"
            )

            # Check if adding this would exceed max_chars
            if total_chars + len(chunk_text) > max_chars:
                # Try to fit a truncated version
                remaining = max_chars - total_chars
                if remaining > 200:  # Only add if we have meaningful space
                    truncated = chunk_text[:remaining] + "...\n"
                    context_parts.append(truncated)
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        context = "\n".join(context_parts)

        logger.debug(
            "context_formatted",
            num_chunks=len(context_parts),
            total_chars=len(context),
        )

        return context


# Singleton instance for convenience
_retriever_instance: Optional[Retriever] = None


async def get_retriever() -> Retriever:
    """Get or create a singleton retriever instance.

    Returns:
        Retriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
        # Pre-load vector store
        await _retriever_instance._ensure_vector_store()
    return _retriever_instance


# Convenience function
async def retrieve_for_query(query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
    """Retrieve relevant chunks for a query (convenience function).

    Args:
        query: User query text
        top_k: Number of results to return

    Returns:
        List of RetrievalResult objects
    """
    retriever = await get_retriever()
    return await retriever.retrieve(query, top_k=top_k)
