"""Unit tests for ToolRouter."""
import pytest
from unittest.mock import patch
from src.tool_router import ToolRouter
from src.models import TaskSpec


def make_task(tool: str, input_str: str = "test", depends_on=None) -> TaskSpec:
    return TaskSpec(
        task_id="t1",
        description="test task",
        tool=tool,
        input=input_str,
        depends_on=depends_on or [],
    )


def test_router_calls_search_tool():
    router = ToolRouter()
    with patch("src.tool_router.search_tool", return_value="search result") as mock_search:
        result = router.route(make_task("search", "AI news"), {})
    assert result.success is True
    assert result.output == "search result"
    mock_search.assert_called_once_with("AI news")


def test_router_resolves_placeholder():
    router = ToolRouter()
    with patch("src.tool_router.summarize_tool", return_value="summary") as mock_sum:
        task = make_task("summarize", "{{t0}}", depends_on=["t0"])
        result = router.route(task, {"t0": "prior output text"})
    mock_sum.assert_called_once_with("prior output text")
    assert result.success is True


def test_router_returns_error_on_tool_exception():
    router = ToolRouter()
    with patch("src.tool_router.search_tool", side_effect=Exception("network error")):
        result = router.route(make_task("search"), {})
    assert result.success is False
    assert "network error" in result.error


def test_router_returns_error_for_unknown_tool():
    router = ToolRouter()
    task = TaskSpec(task_id="t1", description="x", tool="unknown_tool", input="x", depends_on=[])
    result = router.route(task, {})
    assert result.success is False
    assert "Unknown tool" in result.error
