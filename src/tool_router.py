"""Tool router: dispatches TaskSpec objects to the correct tool implementation."""
from __future__ import annotations
import logging
import re
from typing import Callable
from src.models import TaskSpec, ToolResult
from src.tools.search import search_tool
from src.tools.summarize import summarize_tool
from src.tools.file_reader import file_reader_tool
from src.tools.data_analysis import data_analysis_tool

logger = logging.getLogger(__name__)

# Registry mapping tool names to callables
TOOL_REGISTRY: dict[str, Callable[[str], str]] = {
    "search": search_tool,
    "summarize": summarize_tool,
    "read_file": file_reader_tool,
    "analyze": data_analysis_tool,
}

# Pattern to detect {{task_id}} placeholders in task inputs
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class ToolRouter:
    def route(self, task: TaskSpec, prior_results: dict[str, str]) -> ToolResult:
        """
        Execute the tool specified in task, injecting prior results for placeholders.

        Args:
            task: The task to execute.
            prior_results: Map of task_id -> output for completed tasks.

        Returns:
            ToolResult with success/failure status and output.
        """
        tool_fn = TOOL_REGISTRY.get(task.tool)
        if tool_fn is None:
            return ToolResult(
                task_id=task.task_id,
                tool=task.tool,
                success=False,
                error=f"Unknown tool: '{task.tool}'",
            )

        resolved_input = self._resolve_input(task.input, prior_results)

        try:
            output = tool_fn(resolved_input)
            return ToolResult(
                task_id=task.task_id,
                tool=task.tool,
                output=output,
                success=True,
            )
        except Exception as exc:
            logger.error("Tool '%s' failed for task '%s': %s", task.tool, task.task_id, exc)
            return ToolResult(
                task_id=task.task_id,
                tool=task.tool,
                success=False,
                error=str(exc),
            )

    def _resolve_input(self, raw_input: str, prior_results: dict[str, str]) -> str:
        """Replace {{task_id}} placeholders with actual prior task outputs."""
        def replacer(match: re.Match) -> str:
            tid = match.group(1)
            return prior_results.get(tid, f"[missing output for {tid}]")

        return _PLACEHOLDER_RE.sub(replacer, raw_input)
