"""Unit tests for GrokPlanner."""
import json
import pytest
from unittest.mock import MagicMock
from src.planner import GrokPlanner
from src.models import TaskSpec


def make_planner(response: str) -> GrokPlanner:
    grok = MagicMock()
    grok.plan.return_value = response
    return GrokPlanner(grok)


def test_plan_parses_valid_json():
    tasks_json = json.dumps([
        {"task_id": "t1", "description": "Search", "tool": "search", "input": "AI news", "depends_on": []}
    ])
    planner = make_planner(tasks_json)
    tasks = planner.plan("What is AI?", [])
    assert len(tasks) == 1
    assert tasks[0].tool == "search"


def test_plan_strips_markdown_fences():
    tasks_json = '```json\n[{"task_id":"t1","description":"x","tool":"search","input":"q","depends_on":[]}]\n```'
    planner = make_planner(tasks_json)
    tasks = planner.plan("query", [])
    assert tasks[0].task_id == "t1"


def test_plan_falls_back_on_invalid_json():
    grok = MagicMock()
    grok.plan.return_value = "not json at all"
    planner = GrokPlanner(grok)
    tasks = planner.plan("query", [])
    # Should return fallback single search task
    assert len(tasks) == 1
    assert tasks[0].tool == "search"


def test_plan_rejects_invalid_depends_on():
    tasks_json = json.dumps([
        {"task_id": "t1", "description": "x", "tool": "search", "input": "q", "depends_on": ["t99"]}
    ])
    grok = MagicMock()
    grok.plan.return_value = tasks_json
    planner = GrokPlanner(grok)
    # All retries fail validation, falls back to single search
    tasks = planner.plan("query", [])
    assert tasks[0].tool == "search"
    assert tasks[0].depends_on == []
