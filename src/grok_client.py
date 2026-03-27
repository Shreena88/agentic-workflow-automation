"""Groq API client using the OpenAI-compatible Groq endpoint."""
from __future__ import annotations
import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a task planning agent. Given a user query and optional context,
decompose the query into a list of executable tasks. Each task must use one of these tools:
- search: web search via DuckDuckGo
- summarize: summarize a piece of text using LLM
- read_file: read a local file (PDF or text)
- analyze: run pandas data analysis on a CSV file

Respond ONLY with a valid JSON array. Example:
[
  {"task_id": "t1", "description": "Search for X", "tool": "search", "input": "X latest news", "depends_on": []},
  {"task_id": "t2", "description": "Summarize results", "tool": "summarize", "input": "{{t1}}", "depends_on": ["t1"]}
]

Rules:
- Use {{task_id}} as a placeholder in input to inject a prior task's output.
- Keep the plan to 5 tasks or fewer.
- Return ONLY the JSON array, no markdown, no explanation.
"""


class GrokClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        # Read directly from env — dotenv must be loaded before this is called
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.environ.get("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("GROK_API_KEY is not set in environment or .env file")
        # Always default to Groq
        self.base_url = base_url or os.getenv("GROK_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = model or os.getenv("GROK_MODEL", "llama-3.3-70b-versatile")

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        logger.info("GrokClient initialized: base_url=%s model=%s", self.base_url, self.model)

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Send a chat completion request and return the assistant message content."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def plan(self, query: str, context: list[str]) -> str:
        context_block = "\n".join(context) if context else "No prior context."
        user_msg = f"Context:\n{context_block}\n\nQuery: {query}"
        return self.chat(PLANNER_SYSTEM_PROMPT, user_msg)

    def summarize(self, text: str) -> str:
        """Truncate input if it's too large for a single summary call."""
        # Simple limit: ~8000 chars to avoid exceeding context tokens
        max_chars = 30000
        input_text = text
        if len(text) > max_chars:
            input_text = text[:max_chars] + "\n\n[... content truncated for summarization ...]"
            logger.warning("Summarization input truncated (original length: %d)", len(text))

        system = "You are a concise summarization assistant. Summarize the following text clearly."
        return self.chat(system, input_text, temperature=0.3)

    def generate_final_response(self, query: str, results: dict[str, str]) -> str:
        """Compose a final answer from tool results, with heavy truncation to prevent 413 Errors."""
        max_total_chars = 32000
        per_task_limit = max_total_chars // (len(results) or 1)

        safe_results = {}
        for tid, output in results.items():
            if len(output) > per_task_limit:
                safe_results[tid] = output[:per_task_limit] + "\n\n[... truncated for answer generation ...]"
            else:
                safe_results[tid] = output

        results_block = "\n\n".join(f"[{tid}]: {out}" for tid, out in safe_results.items())
        
        system = (
            "You are a helpful AI assistant. Using the tool results below, "
            "provide a comprehensive, well-structured answer to the user's query."
        )
        user_msg = f"Query: {query}\n\nTool Results:\n{results_block}"
        return self.chat(system, user_msg, temperature=0.5)
