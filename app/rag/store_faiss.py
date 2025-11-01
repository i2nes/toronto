"""FAISS vector store for semantic search.

Handles:
- Runtime embedding dimension detection
- FAISS index initialization and loading
- Vector addition and search
- Metadata persistence
"""
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
import structlog

from app import config
from app.llm_client import ollama_client

logger = structlog.get_logger()


class FAISSVectorStore:
    """FAISS-based vector store with dimension detection and metadata."""

    def __init__(
        self,
        index_dir: Path = None,
        embedding_model: str = None,
    ):
        """Initialize the FAISS vector store.

        Args:
            index_dir: Directory to store index and metadata (default: DATA_DIR)
            embedding_model: Embedding model name (default from config)
        """
        self.index_dir = index_dir or config.DATA_DIR
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL

        self.index_path = self.index_dir / "vectors.index"
        self.metadata_path = self.index_dir / "metadata.json"

        self.index: Optional[faiss.Index] = None
        self.dimension: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

        logger.info(
            "faiss_store_initialized",
            index_dir=str(self.index_dir),
            embedding_model=self.embedding_model,
        )

    async def get_embedding_dimension(self) -> int:
        """Detect embedding dimension by embedding a test string.

        Returns:
            Embedding dimension

        Raises:
            RuntimeError: If embedding fails
        """
        logger.info("detecting_embedding_dimension", model=self.embedding_model)

        try:
            response = await ollama_client.embeddings(
                model=self.embedding_model, prompt="test"
            )
            embedding = response.get("embedding", [])

            if not embedding:
                raise RuntimeError("Empty embedding returned from Ollama")

            dimension = len(embedding)
            logger.info("embedding_dimension_detected", dimension=dimension)
            return dimension

        except Exception as e:
            logger.error(
                "embedding_dimension_detection_failed",
                model=self.embedding_model,
                error=str(e),
            )
            raise RuntimeError(
                f"Failed to detect embedding dimension: {e}"
            ) from e

    async def init_new_index(self, dimension: Optional[int] = None) -> None:
        """Initialize a new FAISS index.

        Args:
            dimension: Embedding dimension (auto-detected if not provided)

        Raises:
            RuntimeError: If initialization fails
        """
        if dimension is None:
            dimension = await self.get_embedding_dimension()

        self.dimension = dimension

        # Use IndexFlatL2 (exact search, simple, works for <100k vectors)
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store metadata
        self.metadata = {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.dimension,
            "index_type": "IndexFlatL2",
            "vector_count": 0,
        }

        logger.info(
            "faiss_index_initialized",
            dimension=self.dimension,
            index_type="IndexFlatL2",
        )

    async def load_index(self) -> None:
        """Load existing FAISS index from disk.

        Validates dimension compatibility with current embedding model.

        Raises:
            FileNotFoundError: If index files don't exist
            ValueError: If dimension mismatch detected
            RuntimeError: If loading fails
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        # Load metadata
        try:
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}") from e

        stored_model = self.metadata.get("embedding_model")
        stored_dim = self.metadata.get("embedding_dimension")

        # Detect current model dimension
        current_dim = await self.get_embedding_dimension()

        # Validate dimension compatibility
        if current_dim != stored_dim:
            raise ValueError(
                f"Dimension mismatch: index was built with {stored_model} "
                f"(dim={stored_dim}), but current model {self.embedding_model} "
                f"has dim={current_dim}. Please rebuild the index."
            )

        # Load FAISS index
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = stored_dim
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}") from e

        logger.info(
            "faiss_index_loaded",
            dimension=self.dimension,
            vector_count=self.index.ntotal,
            model=stored_model,
        )

    async def save_index(self) -> None:
        """Save FAISS index and metadata to disk.

        Raises:
            RuntimeError: If save fails
        """
        if self.index is None:
            raise RuntimeError("No index to save. Initialize or load an index first.")

        # Ensure directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Update metadata
        self.metadata["vector_count"] = self.index.ntotal

        # Save FAISS index
        try:
            faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            raise RuntimeError(f"Failed to save FAISS index: {e}") from e

        # Save metadata
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save metadata: {e}") from e

        logger.info(
            "faiss_index_saved",
            index_path=str(self.index_path),
            vector_count=self.index.ntotal,
        )

    async def add_vectors(
        self, embeddings: List[List[float]], vector_ids: Optional[List[int]] = None
    ) -> List[int]:
        """Add vectors to the FAISS index.

        Args:
            embeddings: List of embedding vectors
            vector_ids: Optional list of IDs (auto-assigned if not provided)

        Returns:
            List of vector IDs (0-indexed positions in the index)

        Raises:
            RuntimeError: If no index initialized or dimension mismatch
        """
        if self.index is None:
            raise RuntimeError("No index initialized. Call init_new_index() first.")

        if not embeddings:
            return []

        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)

        # Validate dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        # Get starting ID (current size of index)
        start_id = self.index.ntotal

        # Add to index
        self.index.add(vectors)

        # Generate IDs (sequential)
        if vector_ids is None:
            vector_ids = list(range(start_id, start_id + len(embeddings)))

        logger.info(
            "vectors_added",
            count=len(embeddings),
            total_vectors=self.index.ntotal,
        )

        return vector_ids

    async def search(
        self, query_embedding: List[float], top_k: int = None
    ) -> Tuple[List[int], List[float]]:
        """Search for similar vectors in the index.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return (default from config)

        Returns:
            Tuple of (vector_ids, distances)

        Raises:
            RuntimeError: If no index initialized
        """
        if self.index is None:
            raise RuntimeError("No index initialized. Call load_index() first.")

        if top_k is None:
            top_k = config.RETRIEVAL_TOP_K

        # Convert to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Validate dimension
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {query_vector.shape[1]}"
            )

        # Ensure we don't request more results than we have
        top_k = min(top_k, self.index.ntotal)

        if top_k == 0:
            return [], []

        # Search
        distances, indices = self.index.search(query_vector, top_k)

        # Convert to lists
        vector_ids = indices[0].tolist()
        distance_scores = distances[0].tolist()

        logger.info(
            "vector_search_completed",
            top_k=top_k,
            results_found=len(vector_ids),
        )

        return vector_ids, distance_scores

    async def init_or_load(self) -> None:
        """Initialize new index or load existing one.

        Automatically detects if index exists and loads it, otherwise creates new.

        Raises:
            ValueError: If dimension mismatch on load
            RuntimeError: If initialization or loading fails
        """
        if self.index_path.exists() and self.metadata_path.exists():
            logger.info("existing_index_detected", path=str(self.index_path))
            await self.load_index()
        else:
            logger.info("no_index_found_initializing_new")
            await self.init_new_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        if self.index is None:
            return {
                "initialized": False,
                "vector_count": 0,
                "dimension": None,
            }

        return {
            "initialized": True,
            "vector_count": self.index.ntotal,
            "dimension": self.dimension,
            "embedding_model": self.embedding_model,
            "index_exists_on_disk": self.index_path.exists(),
            "metadata": self.metadata,
        }

    async def rebuild_index(self) -> None:
        """Clear and rebuild the index (for reindexing).

        Raises:
            RuntimeError: If rebuild fails
        """
        logger.warning("rebuilding_index", index_dir=str(self.index_dir))

        # Delete existing files
        if self.index_path.exists():
            self.index_path.unlink()
            logger.info("deleted_existing_index", path=str(self.index_path))

        if self.metadata_path.exists():
            self.metadata_path.unlink()
            logger.info("deleted_existing_metadata", path=str(self.metadata_path))

        # Initialize new index
        await self.init_new_index()


# Singleton instance for convenience
_store_instance: Optional[FAISSVectorStore] = None


async def get_vector_store() -> FAISSVectorStore:
    """Get or create a singleton vector store instance.

    Returns:
        FAISSVectorStore instance

    Note: This loads the index if it exists
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = FAISSVectorStore()
        await _store_instance.init_or_load()
    return _store_instance
