"""Database initialization and helpers for Toronto AI Assistant.

SQLite database for storing:
- Text chunks from indexed notes
- Metadata about indexing runs
- Mapping between FAISS vector IDs and chunk metadata
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import structlog

from app import config

logger = structlog.get_logger()

DB_PATH = config.DATA_DIR / "assistant.sqlite"


def get_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Initialize the database schema.

    Creates tables if they don't exist:
    - index_metadata: tracks indexing runs and configuration
    - chunks: stores text chunks with metadata and FAISS vector IDs
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Table for tracking indexing runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indexed_at TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                chunk_size INTEGER NOT NULL,
                chunk_overlap INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                total_notes INTEGER NOT NULL,
                notes_directory TEXT NOT NULL,
                metadata_json TEXT
            )
        """)

        # Table for text chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector_id INTEGER NOT NULL,
                note_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                heading_context TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(note_path, chunk_index)
            )
        """)

        # Index for fast vector_id lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_vector_id
            ON chunks(vector_id)
        """)

        # Index for note_path lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_note_path
            ON chunks(note_path)
        """)

        conn.commit()
        logger.info("database_initialized", db_path=str(DB_PATH))

    except Exception as e:
        conn.rollback()
        logger.error("database_init_failed", error=str(e))
        raise
    finally:
        conn.close()


def insert_index_metadata(
    embedding_model: str,
    embedding_dimension: int,
    chunk_size: int,
    chunk_overlap: int,
    total_chunks: int,
    total_notes: int,
    notes_directory: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Record a new indexing run.

    Args:
        embedding_model: Name of the embedding model used
        embedding_dimension: Dimension of the embeddings
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
        total_chunks: Total number of chunks indexed
        total_notes: Total number of notes processed
        notes_directory: Path to the notes directory
        metadata: Optional additional metadata as dict

    Returns:
        ID of the inserted metadata row
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO index_metadata (
                indexed_at, embedding_model, embedding_dimension,
                chunk_size, chunk_overlap, total_chunks, total_notes,
                notes_directory, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            embedding_model,
            embedding_dimension,
            chunk_size,
            chunk_overlap,
            total_chunks,
            total_notes,
            notes_directory,
            json.dumps(metadata) if metadata else None,
        ))

        conn.commit()
        row_id = cursor.lastrowid
        logger.info("index_metadata_inserted", id=row_id, total_chunks=total_chunks)
        return row_id

    except Exception as e:
        conn.rollback()
        logger.error("index_metadata_insert_failed", error=str(e))
        raise
    finally:
        conn.close()


def insert_chunk(
    vector_id: int,
    note_path: str,
    chunk_index: int,
    content: str,
    char_start: int,
    char_end: int,
    heading_context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert a text chunk into the database.

    Args:
        vector_id: ID of the vector in FAISS index
        note_path: Path to the source note file
        chunk_index: Index of this chunk within the note
        content: The actual text content of the chunk
        char_start: Starting character position in the original note
        char_end: Ending character position in the original note
        heading_context: Hierarchical heading context (e.g., "# Section > ## Subsection")
        metadata: Optional additional metadata as dict

    Returns:
        ID of the inserted chunk row
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT OR REPLACE INTO chunks (
                vector_id, note_path, chunk_index, content,
                char_start, char_end, heading_context,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vector_id,
            note_path,
            chunk_index,
            content,
            char_start,
            char_end,
            heading_context,
            json.dumps(metadata) if metadata else None,
            datetime.utcnow().isoformat(),
        ))

        conn.commit()
        row_id = cursor.lastrowid
        return row_id

    except Exception as e:
        conn.rollback()
        logger.error("chunk_insert_failed", error=str(e), note_path=note_path)
        raise
    finally:
        conn.close()


def get_chunks_by_vector_ids(vector_ids: List[int]) -> List[Dict[str, Any]]:
    """Retrieve chunks by their FAISS vector IDs.

    Args:
        vector_ids: List of FAISS vector IDs to retrieve

    Returns:
        List of chunk dictionaries with all fields
    """
    if not vector_ids:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    try:
        placeholders = ",".join("?" * len(vector_ids))
        cursor.execute(f"""
            SELECT
                id, vector_id, note_path, chunk_index, content,
                char_start, char_end, heading_context,
                metadata_json, created_at
            FROM chunks
            WHERE vector_id IN ({placeholders})
        """, vector_ids)

        rows = cursor.fetchall()

        # Convert rows to dictionaries
        chunks = []
        for row in rows:
            chunk = dict(row)
            # Parse JSON metadata if present
            if chunk["metadata_json"]:
                chunk["metadata"] = json.loads(chunk["metadata_json"])
            chunks.append(chunk)

        return chunks

    except Exception as e:
        logger.error("chunks_retrieval_failed", error=str(e))
        raise
    finally:
        conn.close()


def get_latest_index_metadata() -> Optional[Dict[str, Any]]:
    """Get the most recent indexing run metadata.

    Returns:
        Dictionary with metadata fields, or None if no index exists
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM index_metadata
            ORDER BY id DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        if row:
            metadata = dict(row)
            # Parse JSON metadata if present
            if metadata["metadata_json"]:
                metadata["metadata"] = json.loads(metadata["metadata_json"])
            return metadata
        return None

    except Exception as e:
        logger.error("index_metadata_retrieval_failed", error=str(e))
        raise
    finally:
        conn.close()


def clear_all_chunks() -> int:
    """Delete all chunks from the database.

    Used when rebuilding the index from scratch.

    Returns:
        Number of chunks deleted
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM chunks")
        conn.commit()

        logger.info("chunks_cleared", count=count)
        return count

    except Exception as e:
        conn.rollback()
        logger.error("chunks_clear_failed", error=str(e))
        raise
    finally:
        conn.close()


def get_chunk_count() -> int:
    """Get the total number of chunks in the database.

    Returns:
        Total chunk count
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        return count

    except Exception as e:
        logger.error("chunk_count_failed", error=str(e))
        raise
    finally:
        conn.close()


# Initialize database on module import if it doesn't exist
if not DB_PATH.exists():
    init_database()
    logger.info("database_auto_initialized", db_path=str(DB_PATH))
