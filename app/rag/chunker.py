"""Text chunking with overlap for RAG pipeline.

Implements character-based chunking to avoid tokenizer dependencies.
"""
from typing import List, Tuple
from dataclasses import dataclass
import structlog

from app import config

logger = structlog.get_logger()


@dataclass
class TextChunk:
    """Represents a chunk of text with position information."""

    content: str
    char_start: int
    char_end: int
    chunk_index: int


class TextChunker:
    """Character-based text chunker with overlap support."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """Initialize the text chunker.

        Args:
            chunk_size: Size of each chunk in characters (default from config)
            chunk_overlap: Overlap between chunks in characters (default from config)
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        # Validate parameters
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.chunk_overlap}) must be less than "
                f"chunk size ({self.chunk_size})"
            )

        logger.info(
            "chunker_initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of TextChunk objects
        """
        if not text:
            return []

        text_length = len(text)

        # Handle text shorter than chunk size
        if text_length <= self.chunk_size:
            logger.debug(
                "text_shorter_than_chunk_size",
                text_length=text_length,
                chunk_size=self.chunk_size,
            )
            return [
                TextChunk(
                    content=text,
                    char_start=0,
                    char_end=text_length,
                    chunk_index=0,
                )
            ]

        chunks = []
        chunk_index = 0
        start = 0

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Extract chunk
            chunk_content = text[start:end]

            # Try to break at sentence or word boundary (not mid-word)
            # Only if we're not at the end of the text
            if end < text_length:
                chunk_content = self._adjust_chunk_boundary(chunk_content, text, end)
                # Recalculate actual end after adjustment
                end = start + len(chunk_content)

            chunks.append(
                TextChunk(
                    content=chunk_content,
                    char_start=start,
                    char_end=end,
                    chunk_index=chunk_index,
                )
            )

            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            chunk_index += 1

            # Prevent infinite loop if overlap calculation fails
            if start <= chunks[-1].char_start:
                start = chunks[-1].char_end

        logger.info(
            "text_chunked",
            text_length=text_length,
            chunk_count=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) // len(chunks),
        )

        return chunks

    def _adjust_chunk_boundary(
        self, chunk_content: str, full_text: str, original_end: int
    ) -> str:
        """Adjust chunk boundary to avoid breaking mid-word or mid-sentence.

        Args:
            chunk_content: Current chunk content
            full_text: Full text being chunked
            original_end: Original end position in full text

        Returns:
            Adjusted chunk content
        """
        # Try to break at sentence boundary (period, !, ?)
        sentence_breaks = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        for break_char in sentence_breaks:
            last_break = chunk_content.rfind(break_char)
            if last_break > len(chunk_content) * 0.7:  # At least 70% through chunk
                return chunk_content[: last_break + len(break_char)]

        # Try to break at paragraph boundary (double newline)
        last_paragraph = chunk_content.rfind("\n\n")
        if last_paragraph > len(chunk_content) * 0.7:
            return chunk_content[: last_paragraph + 2]

        # Try to break at single newline
        last_newline = chunk_content.rfind("\n")
        if last_newline > len(chunk_content) * 0.7:
            return chunk_content[: last_newline + 1]

        # Try to break at word boundary (space)
        last_space = chunk_content.rfind(" ")
        if last_space > len(chunk_content) * 0.8:  # At least 80% through chunk
            return chunk_content[: last_space + 1]

        # If no good break point found, just return original
        return chunk_content

    def get_chunk_stats(self, chunks: List[TextChunk]) -> dict:
        """Get statistics about a set of chunks.

        Args:
            chunks: List of TextChunk objects

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "chunk_count": 0,
                "total_chars": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        chunk_sizes = [len(c.content) for c in chunks]

        return {
            "chunk_count": len(chunks),
            "total_chars": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "overlap": self.chunk_overlap,
        }


# Singleton instance for convenience
_chunker_instance = None


def get_chunker() -> TextChunker:
    """Get a singleton text chunker instance.

    Returns:
        TextChunker instance with default config
    """
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = TextChunker()
    return _chunker_instance


# Convenience function
def chunk_text(text: str) -> List[TextChunk]:
    """Chunk text using default chunker (convenience function).

    Args:
        text: Text to chunk

    Returns:
        List of TextChunk objects
    """
    chunker = get_chunker()
    return chunker.chunk_text(text)
