"""File watcher for automatic note reindexing.

Monitors the notes/ directory for changes and triggers reindexing
for modified, created, or deleted markdown files.
"""
from pathlib import Path
from typing import Optional, Callable, Set
import asyncio
from datetime import datetime, timedelta
import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from app import config, db
from app.rag.ingest import IngestPipeline
from app.rag.store_faiss import FAISSVectorStore

logger = structlog.get_logger()


class MarkdownFileHandler(FileSystemEventHandler):
    """Handler for markdown file system events."""

    def __init__(
        self,
        ingest_pipeline: IngestPipeline,
        vector_store: FAISSVectorStore,
        debounce_seconds: float = 2.0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize the file handler.

        Args:
            ingest_pipeline: Pipeline for ingesting files
            vector_store: Vector store for managing embeddings
            debounce_seconds: Time to wait before processing changes (prevents rapid re-indexing)
            loop: Event loop to use for scheduling async tasks
        """
        super().__init__()
        self.ingest_pipeline = ingest_pipeline
        self.vector_store = vector_store
        self.debounce_seconds = debounce_seconds
        self.loop = loop or asyncio.get_event_loop()

        # Track pending changes with debouncing
        self._pending_changes: Set[Path] = set()
        self._last_change_time: Optional[datetime] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info(
            "markdown_file_handler_initialized",
            debounce_seconds=debounce_seconds,
        )

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            file_path = Path(event.src_path)
            logger.info("file_created", path=str(file_path))
            self._schedule_reindex(file_path, "created")

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            file_path = Path(event.src_path)
            logger.info("file_modified", path=str(file_path))
            self._schedule_reindex(file_path, "modified")

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            file_path = Path(event.src_path)
            logger.info("file_deleted", path=str(file_path))
            self._schedule_deletion(file_path)

    def _schedule_reindex(self, file_path: Path, event_type: str):
        """Schedule a file for reindexing with debouncing.

        Args:
            file_path: Path to the file
            event_type: Type of event (created/modified)
        """
        self._pending_changes.add(file_path)
        self._last_change_time = datetime.now()

        # Start processing task if not already running
        if self._processing_task is None or self._processing_task.done():
            # Schedule coroutine on the main event loop from this thread
            future = asyncio.run_coroutine_threadsafe(
                self._debounced_process(), self.loop
            )
            # Convert future to task for tracking (though we can't access the task directly)
            # We'll rely on the future completing instead
            self._processing_task = None  # Reset since we're using futures now

    def _schedule_deletion(self, file_path: Path):
        """Schedule a file for deletion from index.

        Args:
            file_path: Path to the deleted file
        """
        # Process deletions immediately (no debouncing needed)
        # Schedule coroutine on the main event loop from this thread
        asyncio.run_coroutine_threadsafe(
            self._handle_deletion(file_path), self.loop
        )

    async def _debounced_process(self):
        """Process pending changes after debounce period."""
        while not self._shutdown:
            # Wait for debounce period
            await asyncio.sleep(self.debounce_seconds)

            # Check if more changes came in during debounce
            if self._last_change_time:
                time_since_last_change = datetime.now() - self._last_change_time
                if time_since_last_change < timedelta(seconds=self.debounce_seconds):
                    # More changes came in, wait longer
                    continue

            # Process all pending changes
            if self._pending_changes:
                changes = self._pending_changes.copy()
                self._pending_changes.clear()
                self._last_change_time = None

                await self._reindex_files(changes)
                break

    async def _reindex_files(self, file_paths: Set[Path]):
        """Reindex a set of files.

        Args:
            file_paths: Set of file paths to reindex
        """
        logger.info(
            "reindexing_files",
            count=len(file_paths),
            files=[str(p) for p in file_paths],
        )

        for file_path in file_paths:
            try:
                # Remove old chunks and vectors for this file
                await self._remove_file_from_index(file_path)

                # Ingest the file if it still exists
                if file_path.exists():
                    await self.ingest_pipeline.ingest_file(file_path)
                    logger.info("file_reindexed", path=str(file_path))
                else:
                    logger.warning("file_disappeared", path=str(file_path))

            except Exception as e:
                logger.error(
                    "reindex_failed",
                    path=str(file_path),
                    error=str(e),
                    error_type=type(e).__name__,
                )

        # Save updated index
        await self.vector_store.save_index()
        logger.info("reindex_batch_completed", count=len(file_paths))

    async def _handle_deletion(self, file_path: Path):
        """Handle file deletion by removing from index.

        Args:
            file_path: Path to the deleted file
        """
        logger.info("handling_file_deletion", path=str(file_path))

        try:
            await self._remove_file_from_index(file_path)
            await self.vector_store.save_index()
            logger.info("file_removed_from_index", path=str(file_path))

        except Exception as e:
            logger.error(
                "deletion_handling_failed",
                path=str(file_path),
                error=str(e),
            )

    async def _remove_file_from_index(self, file_path: Path):
        """Remove all chunks and vectors for a file.

        Args:
            file_path: Path to the file (absolute or relative to notes/)
        """
        # Convert to relative path if needed
        try:
            relative_path = file_path.relative_to(config.NOTES_DIR)
        except ValueError:
            # Already relative or not in notes dir
            relative_path = file_path.name if file_path.is_absolute() else file_path

        # Get all chunk IDs for this file
        chunk_ids = db.get_chunk_ids_for_file(str(relative_path))

        if chunk_ids:
            # Remove from vector store
            await self.vector_store.delete_by_ids(chunk_ids)

            # Remove from database
            db.delete_chunks_for_file(str(relative_path))

            logger.info(
                "removed_file_chunks",
                path=str(relative_path),
                chunk_count=len(chunk_ids),
            )
        else:
            logger.debug("no_chunks_found_for_file", path=str(relative_path))

    def shutdown(self):
        """Shutdown the handler and cancel pending tasks."""
        self._shutdown = True
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()


class NotesWatcher:
    """Watcher for the notes/ directory."""

    def __init__(
        self,
        notes_dir: Optional[Path] = None,
        debounce_seconds: float = 2.0,
    ):
        """Initialize the notes watcher.

        Args:
            notes_dir: Directory to watch (default from config)
            debounce_seconds: Debounce period for file changes
        """
        self.notes_dir = notes_dir or config.NOTES_DIR
        self.debounce_seconds = debounce_seconds

        # Initialize components
        self.vector_store = FAISSVectorStore()
        self.ingest_pipeline = None  # Will be created in start() with initialized vector store

        # Event handler and observer will be created in start()
        self.event_handler = None
        self.observer = None

        self._started = False

        logger.info(
            "notes_watcher_initialized",
            notes_dir=str(self.notes_dir),
            debounce_seconds=debounce_seconds,
        )

    async def start(self):
        """Start watching for file changes."""
        if self._started:
            logger.warning("watcher_already_started")
            return

        # Ensure vector store is loaded
        await self.vector_store.init_or_load()

        # Create ingest pipeline with the initialized vector store
        self.ingest_pipeline = IngestPipeline(
            notes_dir=self.notes_dir,
            vector_store=self.vector_store,
        )

        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Create event handler with the event loop
        self.event_handler = MarkdownFileHandler(
            ingest_pipeline=self.ingest_pipeline,
            vector_store=self.vector_store,
            debounce_seconds=self.debounce_seconds,
            loop=loop,
        )

        # Create and start observer
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            str(self.notes_dir),
            recursive=True,
        )
        self.observer.start()
        self._started = True

        logger.info(
            "notes_watcher_started",
            notes_dir=str(self.notes_dir),
        )

    def stop(self):
        """Stop watching for file changes."""
        if not self._started:
            return

        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5.0)

        if self.event_handler:
            self.event_handler.shutdown()

        self._started = False

        logger.info("notes_watcher_stopped")

    def is_alive(self) -> bool:
        """Check if watcher is running.

        Returns:
            True if watcher is active
        """
        return self._started and self.observer is not None and self.observer.is_alive()


# Global watcher instance
_watcher_instance: Optional[NotesWatcher] = None


async def get_watcher() -> NotesWatcher:
    """Get or create the global watcher instance.

    Returns:
        NotesWatcher instance
    """
    global _watcher_instance

    if _watcher_instance is None:
        _watcher_instance = NotesWatcher()
        await _watcher_instance.start()

    return _watcher_instance


async def stop_watcher():
    """Stop the global watcher instance."""
    global _watcher_instance

    if _watcher_instance is not None:
        _watcher_instance.stop()
        _watcher_instance = None
