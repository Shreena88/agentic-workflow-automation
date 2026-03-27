"""Shared Pydantic models used across the system."""
from __future__ import annotations
from typing import Literal, TypedDict
from uuid import uuid4
from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    task_id: str
    description: str
    tool: Literal["search", "summarize", "read_file", "analyze"]
    input: str
    depends_on: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    task_id: str
    tool: str
    output: str = ""
    success: bool = True
    error: str | None = None


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    max_tasks: int = Field(default=5, ge=1, le=10)


class AgentResponse(BaseModel):
    session_id: str
    answer: str
    tasks_executed: list[ToolResult]
    latency_ms: float


class ExecutionResult(BaseModel):
    session_id: str
    task_results: list[ToolResult]
    final_answer: str


# LangGraph state schema
class AgentState(TypedDict):
    query: str
    session_id: str
    context: list[str]
    tasks: list[dict]       # serialized TaskSpec dicts
    results: dict[str, str] # task_id -> output
    final_answer: str
