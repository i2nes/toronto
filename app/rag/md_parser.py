"""Markdown parser for extracting content and metadata from .md files.

Handles:
- YAML frontmatter parsing
- Heading hierarchy extraction
- Clean text extraction
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import yaml
import structlog

logger = structlog.get_logger()


@dataclass
class MarkdownDocument:
    """Parsed markdown document with content and metadata."""

    path: Path
    content: str
    frontmatter: Dict[str, Any]
    headings: List["Heading"]
    text_without_frontmatter: str


@dataclass
class Heading:
    """Represents a markdown heading with hierarchy."""

    level: int  # 1-6 for h1-h6
    text: str
    char_position: int
    line_number: int


class MarkdownParser:
    """Parser for markdown documents with frontmatter support."""

    # Regex for YAML frontmatter (must be at start of file)
    FRONTMATTER_PATTERN = re.compile(
        r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE
    )

    # Regex for markdown headings
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)

    def __init__(self):
        """Initialize the markdown parser."""
        pass

    def parse_file(self, file_path: Path) -> MarkdownDocument:
        """Parse a markdown file and extract content and metadata.

        Args:
            file_path: Path to the markdown file

        Returns:
            MarkdownDocument with parsed content

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            logger.error("markdown_encoding_error", path=str(file_path), error=str(e))
            raise

        # Parse frontmatter
        frontmatter, text_without_frontmatter = self._parse_frontmatter(content)

        # Extract headings
        headings = self._extract_headings(text_without_frontmatter)

        logger.info(
            "markdown_parsed",
            path=str(file_path),
            has_frontmatter=bool(frontmatter),
            heading_count=len(headings),
            content_length=len(text_without_frontmatter),
        )

        return MarkdownDocument(
            path=file_path,
            content=content,
            frontmatter=frontmatter,
            headings=headings,
            text_without_frontmatter=text_without_frontmatter,
        )

    def _parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from markdown content.

        Args:
            content: Full markdown content

        Returns:
            Tuple of (frontmatter_dict, content_without_frontmatter)
        """
        match = self.FRONTMATTER_PATTERN.match(content)

        if not match:
            # No frontmatter
            return {}, content

        # Extract and parse YAML
        yaml_content = match.group(1)
        try:
            frontmatter = yaml.safe_load(yaml_content)
            if frontmatter is None:
                frontmatter = {}
        except yaml.YAMLError as e:
            logger.warning(
                "frontmatter_parse_error",
                error=str(e),
                yaml_preview=yaml_content[:100],
            )
            frontmatter = {}

        # Remove frontmatter from content
        content_without_frontmatter = content[match.end() :]

        return frontmatter, content_without_frontmatter

    def _extract_headings(self, content: str) -> List[Heading]:
        """Extract all markdown headings with their positions.

        Args:
            content: Markdown content (without frontmatter)

        Returns:
            List of Heading objects
        """
        headings = []

        for match in self.HEADING_PATTERN.finditer(content):
            level = len(match.group(1))  # Count # symbols
            text = match.group(2).strip()
            char_position = match.start()

            # Calculate line number
            line_number = content[: match.start()].count("\n") + 1

            headings.append(
                Heading(
                    level=level,
                    text=text,
                    char_position=char_position,
                    line_number=line_number,
                )
            )

        return headings

    def get_heading_context(
        self, headings: List[Heading], char_position: int
    ) -> str:
        """Get hierarchical heading context for a given character position.

        Returns a breadcrumb-like string of headings leading to this position.

        Args:
            headings: List of all headings in the document
            char_position: Character position to get context for

        Returns:
            Heading context string like "# Main > ## Sub > ### Detail"
        """
        if not headings:
            return ""

        # Find all headings before this position
        preceding_headings = [h for h in headings if h.char_position < char_position]

        if not preceding_headings:
            return ""

        # Build hierarchical context
        context_stack = []
        current_level = 0

        for heading in preceding_headings:
            # If we go deeper, add to stack
            if heading.level > current_level:
                context_stack.append(heading)
                current_level = heading.level
            # If we go back up or stay same level, pop and add
            else:
                # Pop headings at same or deeper level
                while context_stack and context_stack[-1].level >= heading.level:
                    context_stack.pop()
                context_stack.append(heading)
                current_level = heading.level

        # Build context string
        context_parts = []
        for heading in context_stack:
            prefix = "#" * heading.level
            context_parts.append(f"{prefix} {heading.text}")

        return " > ".join(context_parts)

    def get_metadata_for_chunk(
        self,
        doc: MarkdownDocument,
        chunk_start: int,
        chunk_end: int,
    ) -> Dict[str, Any]:
        """Get metadata for a specific chunk of the document.

        Args:
            doc: Parsed markdown document
            chunk_start: Starting character position in text_without_frontmatter
            chunk_end: Ending character position in text_without_frontmatter

        Returns:
            Dictionary with metadata including heading context
        """
        heading_context = self.get_heading_context(doc.headings, chunk_start)

        metadata = {
            "file_path": str(doc.path),
            "file_name": doc.path.name,
            "heading_context": heading_context,
            "char_start": chunk_start,
            "char_end": chunk_end,
        }

        # Add relevant frontmatter fields
        if doc.frontmatter:
            # Common fields to include
            for field in ["title", "tags", "created", "updated", "author"]:
                if field in doc.frontmatter:
                    value = doc.frontmatter[field]
                    # Convert date/datetime objects to ISO format strings
                    if hasattr(value, "isoformat"):
                        value = value.isoformat()
                    metadata[field] = value

        return metadata


# Singleton instance for convenience
_parser_instance = None


def get_parser() -> MarkdownParser:
    """Get a singleton markdown parser instance.

    Returns:
        MarkdownParser instance
    """
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = MarkdownParser()
    return _parser_instance


# Convenience function
def parse_markdown_file(file_path: Path) -> MarkdownDocument:
    """Parse a markdown file (convenience function).

    Args:
        file_path: Path to markdown file

    Returns:
        Parsed MarkdownDocument
    """
    parser = get_parser()
    return parser.parse_file(file_path)
