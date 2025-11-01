"""Ingest pipeline for indexing markdown notes.

Orchestrates:
- File discovery
- Markdown parsing
- Text chunking
- Embedding generation
- Vector and metadata storage
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import structlog

from app import config, db
from app.llm_client import ollama_client
from app.rag.md_parser import MarkdownParser
from app.rag.chunker import TextChunker
from app.rag.store_faiss import FAISSVectorStore

logger = structlog.get_logger()


class IngestPipeline:
    """Pipeline for ingesting markdown notes into the RAG system."""

    def __init__(
        self,
        notes_dir: Path = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        batch_size: int = 10,
    ):
        """Initialize the ingest pipeline.

        Args:
            notes_dir: Directory containing markdown notes (default from config)
            embedding_model: Embedding model name (default from config)
            chunk_size: Chunk size in characters (default from config)
            chunk_overlap: Chunk overlap in characters (default from config)
            batch_size: Number of embeddings to generate in parallel
        """
        self.notes_dir = notes_dir or config.NOTES_DIR
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.batch_size = batch_size

        self.parser = MarkdownParser()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = FAISSVectorStore(embedding_model=embedding_model)

        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
        }

        logger.info(
            "ingest_pipeline_initialized",
            notes_dir=str(self.notes_dir),
            embedding_model=self.embedding_model,
            chunk_size=self.chunker.chunk_size,
            chunk_overlap=self.chunker.chunk_overlap,
        )

    def discover_markdown_files(self) -> List[Path]:
        """Discover all markdown files in the notes directory.

        Returns:
            List of markdown file paths

        Raises:
            FileNotFoundError: If notes directory doesn't exist
        """
        if not self.notes_dir.exists():
            raise FileNotFoundError(f"Notes directory not found: {self.notes_dir}")

        # Find all .md files recursively
        md_files = list(self.notes_dir.rglob("*.md"))

        logger.info(
            "markdown_files_discovered",
            count=len(md_files),
            notes_dir=str(self.notes_dir),
        )

        return md_files

    async def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return []

        embeddings = []

        # Process in smaller batches to avoid overwhelming Ollama
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Generate embeddings for batch (sequentially for now)
            # TODO: Could make this truly parallel with asyncio.gather
            batch_embeddings = []
            for text in batch:
                try:
                    response = await ollama_client.embeddings(
                        model=self.embedding_model, prompt=text
                    )
                    embedding = response.get("embedding", [])

                    if not embedding:
                        raise RuntimeError(f"Empty embedding returned for text")

                    batch_embeddings.append(embedding)
                    self.stats["embeddings_generated"] += 1

                except Exception as e:
                    logger.error(
                        "embedding_generation_failed",
                        text_preview=text[:100],
                        error=str(e),
                    )
                    raise RuntimeError(f"Failed to generate embedding: {e}") from e

            embeddings.extend(batch_embeddings)

            logger.debug(
                "embeddings_batch_generated",
                batch_size=len(batch),
                total_so_far=len(embeddings),
            )

        return embeddings

    async def ingest_file(self, file_path: Path) -> Dict[str, Any]:
        """Ingest a single markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            Dictionary with ingestion results (chunks_created, etc.)

        Raises:
            Exception: If ingestion fails
        """
        logger.info("ingesting_file", path=str(file_path))

        # Parse markdown
        doc = self.parser.parse_file(file_path)

        # Chunk text
        chunks = self.chunker.chunk_text(doc.text_without_frontmatter)

        if not chunks:
            logger.warning("no_chunks_created", path=str(file_path))
            return {"chunks_created": 0}

        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.generate_embeddings_batch(chunk_texts)

        # Add vectors to FAISS
        vector_ids = await self.vector_store.add_vectors(embeddings)

        # Store chunks in SQLite
        for chunk, vector_id in zip(chunks, vector_ids):
            # Get heading context for this chunk
            heading_context = self.parser.get_heading_context(
                doc.headings, chunk.char_start
            )

            # Get full metadata
            metadata = self.parser.get_metadata_for_chunk(
                doc, chunk.char_start, chunk.char_end
            )

            # Insert into database
            db.insert_chunk(
                vector_id=vector_id,
                note_path=str(file_path.relative_to(self.notes_dir)),
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                heading_context=heading_context,
                metadata=metadata,
            )

        self.stats["chunks_created"] += len(chunks)
        self.stats["files_processed"] += 1

        logger.info(
            "file_ingested",
            path=str(file_path),
            chunks_created=len(chunks),
        )

        return {
            "file_path": str(file_path),
            "chunks_created": len(chunks),
            "embeddings_generated": len(embeddings),
        }

    async def ingest_all(
        self, rebuild: bool = False, progress_callback=None
    ) -> Dict[str, Any]:
        """Ingest all markdown files in the notes directory.

        Args:
            rebuild: If True, clear existing index and rebuild from scratch
            progress_callback: Optional callback function(current, total, file_path)

        Returns:
            Dictionary with ingestion statistics

        Raises:
            Exception: If ingestion fails
        """
        logger.info("starting_ingest_all", rebuild=rebuild)

        # Initialize or load vector store
        if rebuild:
            await self.vector_store.rebuild_index()
            db.clear_all_chunks()
            logger.info("index_and_database_cleared")
        else:
            await self.vector_store.init_or_load()

        # Discover files
        md_files = self.discover_markdown_files()

        if not md_files:
            logger.warning("no_markdown_files_found", notes_dir=str(self.notes_dir))
            return self.stats

        # Reset stats
        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
        }

        # Process each file
        for idx, file_path in enumerate(md_files, 1):
            try:
                if progress_callback:
                    progress_callback(idx, len(md_files), file_path)

                await self.ingest_file(file_path)

            except Exception as e:
                logger.error(
                    "file_ingestion_failed",
                    path=str(file_path),
                    error=str(e),
                )
                self.stats["files_failed"] += 1
                # Continue with next file instead of failing entirely

        # Save FAISS index and metadata
        await self.vector_store.save_index()

        # Record index metadata in database
        db.insert_index_metadata(
            embedding_model=self.embedding_model,
            embedding_dimension=self.vector_store.dimension,
            chunk_size=self.chunker.chunk_size,
            chunk_overlap=self.chunker.chunk_overlap,
            total_chunks=self.stats["chunks_created"],
            total_notes=self.stats["files_processed"],
            notes_directory=str(self.notes_dir),
            metadata={
                "files_failed": self.stats["files_failed"],
                "embeddings_generated": self.stats["embeddings_generated"],
            },
        )

        logger.info("ingest_all_completed", stats=self.stats)

        return self.stats


# Convenience function for quick ingestion
async def ingest_notes(rebuild: bool = False) -> Dict[str, Any]:
    """Ingest all markdown notes (convenience function).

    Args:
        rebuild: If True, rebuild index from scratch

    Returns:
        Ingestion statistics
    """
    pipeline = IngestPipeline()
    return await pipeline.ingest_all(rebuild=rebuild)
