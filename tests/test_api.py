"""Integration tests for the FastAPI /run-agent endpoint."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.models import ExecutionResult, ToolResult


@pytest.fixture
def client():
    # Patch engine initialization so no real API keys are needed
    mock_result = ExecutionResult(
        session_id="test-session",
        task_results=[
            ToolResult(task_id="t1", tool="search", output="search result", success=True)
        ],
        final_answer="This is the final answer.",
    )
    mock_engine = MagicMock()
    mock_engine.run.return_value = mock_result

    with patch("src.api._engine", mock_engine):
        from src.api import app
        with TestClient(app) as c:
            yield c, mock_engine


def test_run_agent_success(client):
    c, mock_engine = client
    response = c.post("/run-agent", json={"query": "What is LangGraph?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is the final answer."
    assert len(data["tasks_executed"]) == 1
    assert "session_id" in data
    assert data["latency_ms"] >= 0


def test_run_agent_empty_query_rejected(client):
    c, _ = client
    response = c.post("/run-agent", json={"query": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_run_agent_max_tasks_out_of_range(client):
    c, _ = client
    response = c.post("/run-agent", json={"query": "test", "max_tasks": 99})
    assert response.status_code == 422


def test_health_endpoint(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
