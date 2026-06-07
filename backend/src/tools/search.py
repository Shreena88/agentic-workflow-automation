"""DuckDuckGo web search tool."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

MAX_RESULTS = 5


def search_tool(query: str) -> str:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_RESULTS))

        if not results:
            return f"No results found for: {query}"

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"{i}. {title}\n   {body}\n   URL: {href}")

        return "\n\n".join(lines)

    except Exception as exc:
        logger.error("Search tool error: %s", exc)
        raise
