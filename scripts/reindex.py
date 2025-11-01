#!/usr/bin/env python
"""Reindex markdown notes for RAG pipeline.

Usage:
    python scripts/reindex.py              # Incremental reindex
    python scripts/reindex.py --rebuild    # Full rebuild from scratch
    python scripts/reindex.py --verbose    # Show detailed progress
"""
import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import config
from app.rag.ingest import IngestPipeline
import structlog

logger = structlog.get_logger()


class ProgressReporter:
    """Simple progress reporter for CLI."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = None

    def start(self, message: str):
        """Start progress reporting."""
        self.start_time = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"  {message}")
        print(f"{'=' * 60}\n")

    def update(self, current: int, total: int, file_path: Path):
        """Update progress."""
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        file_name = file_path.name
        print(
            f"\r  [{bar}] {percentage:5.1f}% ({current}/{total}) {file_name[:30]:<30}",
            end="",
            flush=True,
        )

        if self.verbose:
            print()  # New line for verbose mode

    def finish(self, stats: dict):
        """Finish progress reporting."""
        print("\n")  # New line after progress bar
        elapsed = datetime.now() - self.start_time
        elapsed_seconds = elapsed.total_seconds()

        print(f"{'=' * 60}")
        print(f"  Indexing Complete!")
        print(f"{'=' * 60}\n")
        print(f"  📁 Files processed:      {stats['files_processed']}")
        print(f"  ❌ Files failed:         {stats['files_failed']}")
        print(f"  📝 Chunks created:       {stats['chunks_created']}")
        print(f"  🧮 Embeddings generated: {stats['embeddings_generated']}")
        print(f"  ⏱️  Time elapsed:         {elapsed_seconds:.1f}s")

        if stats['chunks_created'] > 0:
            rate = stats['chunks_created'] / elapsed_seconds
            print(f"  ⚡ Indexing rate:        {rate:.1f} chunks/sec")

        print(f"\n{'=' * 60}\n")

        # Check for failures
        if stats['files_failed'] > 0:
            print(f"⚠️  Warning: {stats['files_failed']} file(s) failed to index.")
            print(f"   Check logs for details.\n")

        # Success message
        if stats['files_processed'] > 0:
            print(f"✅ Index ready at: {config.DATA_DIR}/vectors.index")
            print(f"✅ Database at: {config.DATA_DIR}/assistant.sqlite\n")


async def main():
    """Main entry point for reindex script."""
    parser = argparse.ArgumentParser(
        description="Reindex markdown notes for RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/reindex.py              # Incremental reindex
  python scripts/reindex.py --rebuild    # Full rebuild from scratch
  python scripts/reindex.py --verbose    # Show detailed progress
        """,
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from scratch (clears existing data)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose progress output",
    )

    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=None,
        help=f"Notes directory (default: {config.NOTES_DIR})",
    )

    args = parser.parse_args()

    # Create progress reporter
    progress = ProgressReporter(verbose=args.verbose)

    try:
        # Display configuration
        print("\n📋 Configuration:")
        print(f"   Notes directory:  {args.notes_dir or config.NOTES_DIR}")
        print(f"   Embedding model:  {config.EMBEDDING_MODEL}")
        print(f"   Chunk size:       {config.CHUNK_SIZE} chars")
        print(f"   Chunk overlap:    {config.CHUNK_OVERLAP} chars")
        print(f"   Top-K retrieval:  {config.RETRIEVAL_TOP_K}")

        if args.rebuild:
            print("\n⚠️  Rebuild mode: Will clear existing index and database!")
            print("   Press Ctrl+C within 3 seconds to cancel...")
            await asyncio.sleep(3)

        # Start indexing
        action = "Rebuilding" if args.rebuild else "Indexing"
        progress.start(f"{action} Notes")

        # Create pipeline
        pipeline = IngestPipeline(notes_dir=args.notes_dir)

        # Define progress callback
        def on_progress(current, total, file_path):
            progress.update(current, total, file_path)

        # Run ingestion
        stats = await pipeline.ingest_all(
            rebuild=args.rebuild,
            progress_callback=on_progress if not args.verbose else None,
        )

        # Show results
        progress.finish(stats)

        # Exit with error code if there were failures
        if stats["files_failed"] > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Indexing cancelled by user.\n")
        sys.exit(1)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        logger.error("reindex_script_failed", error=str(e), error_type=type(e).__name__)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
