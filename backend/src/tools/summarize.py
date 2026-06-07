"""Grok-powered summarization tool."""
from __future__ import annotations
import logging
from src.grok_client import GrokClient

logger = logging.getLogger(__name__)

# Module-level client; injected at app startup via set_client()
_grok: GrokClient | None = None


def set_client(client: GrokClient) -> None:
    global _grok
    _grok = client


def summarize_tool(text: str) -> str:
    """
    Summarize the provided text using Grok LLM.

    Args:
        text: The text to summarize.

    Returns:
        A concise summary string.
    """
    if _grok is None:
        raise RuntimeError("GrokClient not initialized. Call set_client() first.")
    if not text.strip():
        return "Nothing to summarize."
    return _grok.summarize(text)
