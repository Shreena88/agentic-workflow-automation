"""Grok-powered task planner: converts a user query into a list of TaskSpec objects."""
from __future__ import annotations
import json
import logging
import time
from src.grok_client import GrokClient
from src.models import TaskSpec

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds


class GrokPlanner:
    def __init__(self, grok: GrokClient) -> None:
        self.grok = grok

    def plan(self, query: str, context: list[str]) -> list[TaskSpec]:
        """
        Decompose query into tasks. Retries up to MAX_RETRIES on parse failure.
        Falls back to a single search task if all retries fail.
        """
        for attempt in range(MAX_RETRIES):
            try:
                raw = self.grok.plan(query, context)
                tasks = self._parse(raw)
                if tasks:
                    return self._validate(tasks)
            except Exception as exc:
                logger.warning("Planner attempt %d failed: %s", attempt + 1, exc)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        logger.error("All planner retries failed; using fallback single-search plan.")
        return [
            TaskSpec(
                task_id="t1",
                description=f"Search for: {query}",
                tool="search",
                input=query,
                depends_on=[],
            )
        ]

    def _parse(self, raw: str) -> list[TaskSpec]:
        """Extract JSON array from LLM response and parse into TaskSpec list."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array from planner")
        return [TaskSpec(**item) for item in data]

    def _validate(self, tasks: list[TaskSpec]) -> list[TaskSpec]:
        """Ensure all depends_on references are valid task_ids within the plan."""
        ids = {t.task_id for t in tasks}
        for task in tasks:
            for dep in task.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Task {task.task_id} depends on unknown task_id '{dep}'"
                    )
        return tasks
