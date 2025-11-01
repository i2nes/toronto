"""Web search tool using DuckDuckGo HTML scraping."""
import httpx
import re
from typing import List
from pydantic import BaseModel, Field
import structlog

from app.tools.registry import Tool, register_tool
from app import config

logger = structlog.get_logger()


class SearchInput(BaseModel):
    """Input for web search tool."""
    query: str = Field(..., description="Search query", max_length=200)
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=10
    )


class SearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    snippet: str


class SearchOutput(BaseModel):
    """Output from web search tool."""
    query: str
    results: List[SearchResult]
    result_count: int


async def search_handler(input_data: SearchInput) -> SearchOutput:
    """Perform web search using DuckDuckGo.

    Args:
        input_data: SearchInput with query and max_results

    Returns:
        SearchOutput with search results
    """
    query = input_data.query
    max_results = min(input_data.max_results, getattr(config, 'SEARCH_MAX_RESULTS', 10))

    logger.info("web_search_started", query=query, max_results=max_results)

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # DuckDuckGo HTML search
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; TorontoAI/1.0)"
                },
                timeout=10.0,
            )
            response.raise_for_status()

        # Parse results from HTML
        results = _parse_duckduckgo_html(response.text, max_results)

        # Filter by allowed/blocked domains if configured
        results = _filter_by_domains(results)

        # Sanitize results
        results = [_sanitize_result(r) for r in results]

        logger.info(
            "web_search_completed",
            query=query,
            result_count=len(results),
        )

        return SearchOutput(
            query=query,
            results=results,
            result_count=len(results),
        )

    except Exception as e:
        logger.error("web_search_failed", query=query, error=str(e))
        # Return empty results on error
        return SearchOutput(
            query=query,
            results=[],
            result_count=0,
        )


def _parse_duckduckgo_html(html: str, max_results: int) -> List[SearchResult]:
    """Parse DuckDuckGo HTML results.

    Args:
        html: HTML response from DuckDuckGo
        max_results: Maximum number of results to extract

    Returns:
        List of SearchResult objects
    """
    results = []

    # Find result blocks
    # DuckDuckGo HTML structure: results are in divs with class "result"
    result_pattern = r'<div class="result[^"]*"[^>]*>.*?<a rel="nofollow" class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>'

    matches = re.findall(result_pattern, html, re.DOTALL)

    for url, title, snippet in matches[:max_results]:
        # Clean HTML from title and snippet
        title_clean = _clean_html(title)
        snippet_clean = _clean_html(snippet)

        if title_clean and url:
            results.append(SearchResult(
                title=title_clean,
                url=url,
                snippet=snippet_clean or "(No description)",
            ))

    return results


def _clean_html(text: str) -> str:
    """Remove HTML tags and decode entities.

    Args:
        text: HTML text to clean

    Returns:
        Clean text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#x27;', "'")
    text = text.replace('&nbsp;', ' ')

    # Clean whitespace
    text = ' '.join(text.split())

    return text.strip()


def _filter_by_domains(results: List[SearchResult]) -> List[SearchResult]:
    """Filter results by allowed/blocked domains from config.

    Args:
        results: List of search results

    Returns:
        Filtered list of results
    """
    allowed_domains = getattr(config, 'SEARCH_ALLOWED_DOMAINS', None)
    blocked_domains = getattr(config, 'SEARCH_BLOCKED_DOMAINS', set())

    if not allowed_domains and not blocked_domains:
        return results

    filtered = []
    for result in results:
        domain = _extract_domain(result.url)

        # Check blocked domains
        if domain in blocked_domains:
            logger.debug("result_blocked", domain=domain, url=result.url)
            continue

        # Check allowed domains (if configured)
        if allowed_domains and domain not in allowed_domains:
            logger.debug("result_not_allowed", domain=domain, url=result.url)
            continue

        filtered.append(result)

    return filtered


def _extract_domain(url: str) -> str:
    """Extract domain from URL.

    Args:
        url: Full URL

    Returns:
        Domain name
    """
    match = re.match(r'https?://([^/]+)', url)
    if match:
        return match.group(1).lower()
    return ""


def _sanitize_result(result: SearchResult) -> SearchResult:
    """Sanitize search result to prevent injection attacks.

    Args:
        result: Search result to sanitize

    Returns:
        Sanitized result
    """
    # Limit lengths
    title = result.title[:200]
    snippet = result.snippet[:500]
    url = result.url[:500]

    # Remove any control characters
    title = ''.join(char for char in title if ord(char) >= 32 or char in '\n\r\t')
    snippet = ''.join(char for char in snippet if ord(char) >= 32 or char in '\n\r\t')

    return SearchResult(
        title=title,
        url=url,
        snippet=snippet,
    )


# Create and register the search tool
search_tool = Tool(
    name="search",
    description="Search the web using DuckDuckGo. Returns titles, URLs, and snippets for relevant results.",
    input_model=SearchInput,
    output_model=SearchOutput,
    handler=search_handler,
)

register_tool(search_tool)
