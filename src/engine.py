"""LangGraph-based execution engine that orchestrates the agent workflow."""
from __future__ import annotations
import logging
from langgraph.graph import StateGraph, END
from src.models import AgentState, TaskSpec, ToolResult, ExecutionResult
from src.planner import GrokPlanner
from src.tool_router import ToolRouter
from src.memory import FAISSMemoryStore
from src.grok_client import GrokClient

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(
        self,
        planner: GrokPlanner,
        router: ToolRouter,
        memory: FAISSMemoryStore,
        grok: GrokClient,
        max_tasks: int = 5,
    ) -> None:
        self.planner = planner
        self.router = router
        self.memory = memory
        self.grok = grok
        self.max_tasks = max_tasks

    def run(self, query: str, session_id: str, max_tasks: int | None = None) -> ExecutionResult:
        """Build and execute the LangGraph workflow for the given query."""
        limit = max_tasks or self.max_tasks
        graph = self._build_graph()
        compiled = graph.compile()

        initial_state: AgentState = {
            "query": query,
            "session_id": session_id,
            "context": [],
            "tasks": [],
            "results": {},
            "final_answer": "",
        }

        final_state = compiled.invoke(initial_state)

        task_results = [
            ToolResult(
                task_id=tid,
                tool=self._get_tool_for_task(final_state["tasks"], tid),
                output=output,
                success=True,
            )
            for tid, output in final_state["results"].items()
        ]

        return ExecutionResult(
            session_id=session_id,
            task_results=task_results,
            final_answer=final_state["final_answer"],
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("retrieve_context", self._node_retrieve_context)
        graph.add_node("plan", self._node_plan)
        graph.add_node("execute_tasks", self._node_execute_tasks)
        graph.add_node("generate_response", self._node_generate_response)

        graph.set_entry_point("retrieve_context")
        graph.add_edge("retrieve_context", "plan")
        graph.add_edge("plan", "execute_tasks")
        graph.add_edge("execute_tasks", "generate_response")
        graph.add_edge("generate_response", END)

        return graph

    # --- Graph Nodes ---

    def _node_retrieve_context(self, state: AgentState) -> AgentState:
        context = self.memory.retrieve(state["session_id"], state["query"], top_k=5)
        return {**state, "context": context}

    def _node_plan(self, state: AgentState) -> AgentState:
        tasks = self.planner.plan(state["query"], state["context"])
        tasks = tasks[: self.max_tasks]
        return {**state, "tasks": [t.model_dump() for t in tasks]}

    def _node_execute_tasks(self, state: AgentState) -> AgentState:
        tasks = [TaskSpec(**t) for t in state["tasks"]]
        ordered = _topological_sort(tasks)
        results: dict[str, str] = {}

        for task in ordered:
            result = self.router.route(task, results)
            output = result.output if result.success else f"[ERROR: {result.error}]"
            results[task.task_id] = output
            self.memory.store(
                state["session_id"],
                output,
                {"task_id": task.task_id, "tool": task.tool},
            )

        return {**state, "results": results}

    def _node_generate_response(self, state: AgentState) -> AgentState:
        answer = self.grok.generate_final_response(state["query"], state["results"])
        return {**state, "final_answer": answer}

    def _get_tool_for_task(self, tasks_dicts: list[dict], task_id: str) -> str:
        for t in tasks_dicts:
            if t["task_id"] == task_id:
                return t["tool"]
        return "unknown"


def _topological_sort(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """Kahn's algorithm for topological ordering of tasks by depends_on."""
    id_map = {t.task_id: t for t in tasks}
    in_degree: dict[str, int] = {t.task_id: 0 for t in tasks}
    dependents: dict[str, list[str]] = {t.task_id: [] for t in tasks}

    for task in tasks:
        for dep in task.depends_on:
            in_degree[task.task_id] += 1
            dependents[dep].append(task.task_id)

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    ordered: list[TaskSpec] = []

    while queue:
        tid = queue.pop(0)
        ordered.append(id_map[tid])
        for dependent in dependents[tid]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(tasks):
        raise ValueError("Circular dependency detected in task plan.")

    return ordered
