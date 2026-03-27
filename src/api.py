from __future__ import annotations
import time
import logging
import os
from contextlib import asynccontextmanager
from typing import Any # Added this import
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form # Modified this line to add Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from src.models import AgentRequest, AgentResponse
from src.grok_client import GrokClient
from src.planner import GrokPlanner
from src.tool_router import ToolRouter
from src.memory import FAISSMemoryStore
from src.engine import ExecutionEngine
import src.tools.summarize as summarize_module
from src.tools.visual_analyzer import VisualAnalyzer

load_dotenv(override=True)  # force reload even if already set
logger = logging.getLogger(__name__)

# --- Rate limiter ---
limiter = Limiter(key_func=get_remote_address)

# --- App-level singletons (initialized at startup) ---
_engine: ExecutionEngine | None = None
_memory: FAISSMemoryStore | None = None
_grok: GrokClient | None = None
_visual_analyzer: VisualAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _memory, _grok, _visual_analyzer
    _grok = GrokClient()
    _memory = FAISSMemoryStore(persist_dir=os.getenv("MEMORY_PERSIST_DIR", "./memory"))
    _visual_analyzer = VisualAnalyzer(_grok)
    planner = GrokPlanner(_grok)
    router = ToolRouter()
    summarize_module.set_client(_grok)  # inject Grok into summarize tool
    _engine = ExecutionEngine(
        planner=planner,
        router=router,
        memory=_memory,
        grok=_grok,
        max_tasks=int(os.getenv("MAX_TASKS", "5")),
    )
    logger.info("Agent engine initialized.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Agentic Workflow Automation API",
    version="1.0.0",
    description="LLM-powered agentic system with tool use and FAISS memory.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/run-agent", response_model=AgentResponse)
@limiter.limit("10/minute")
async def run_agent(request: Request, body: AgentRequest) -> AgentResponse:
    """
    Execute the agentic workflow for the given query.

    - Retrieves relevant memory context
    - Plans tasks via Grok LLM
    - Executes tools via LangGraph
    - Returns synthesized final answer
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    start = time.perf_counter()
    try:
        result = _engine.run(
            query=body.query,
            session_id=body.session_id,
            max_tasks=body.max_tasks,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Agent execution failed: %s", exc)
        raise HTTPException(status_code=500, detail="Agent execution failed.")

    latency_ms = (time.perf_counter() - start) * 1000

    return AgentResponse(
        session_id=result.session_id,
        answer=result.final_answer,
        tasks_executed=result.task_results,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    """Return all stored memory texts for a session."""
    import pickle
    from pathlib import Path
    mem_dir = Path(os.getenv("MEMORY_PERSIST_DIR", "./memory")) / session_id
    texts_path = mem_dir / "texts.pkl"
    if not texts_path.exists():
        return {"session_id": session_id, "count": 0, "entries": []}
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)
    return {
        "session_id": session_id,
        "count": len(texts),
        "entries": [{"index": i, "preview": t[:200]} for i, t in enumerate(texts)]
    }


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear all stored memory for a session."""
    import shutil
    from pathlib import Path
    mem_dir = Path(os.getenv("MEMORY_PERSIST_DIR", "./memory")) / session_id
    if mem_dir.exists():
        shutil.rmtree(mem_dir)
    return {"session_id": session_id, "cleared": True}


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(None)):
    """
    Upload a file (PDF, TXT, MD, CSV, JSON) to the data directory
    so it can be referenced by the file_reader tool.
    """
    allowed_extensions = {".pdf", ".txt", ".md", ".csv", ".json"}
    suffix = os.path.splitext(file.filename or "")[-1].lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed_extensions)}"
        )

    data_dir = os.getenv("ALLOWED_FILE_DIRS", "./data")
    os.makedirs(data_dir, exist_ok=True)

    save_path = os.path.join(data_dir, file.filename)
    contents = await file.read()

    with open(save_path, "wb") as f:
        f.write(contents)

    logger.info("File uploaded: %s (%d bytes)", save_path, len(contents))
    
    # Store the fact that this file was uploaded in the session memory
    if session_id and _memory:
        _memory.store(session_id, f"Uploaded file: {save_path}")

    return {
        "filename": file.filename,
        "saved_path": save_path,
        "size_bytes": len(contents),
        "message": f"File saved. Use 'read_file' tool with path: {save_path}"
    }


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Modular CSV analysis using the VisualAnalyzer tool.
    """
    if not _visual_analyzer:
        raise HTTPException(status_code=503, detail="Visual Analyzer not initialized.")

    contents = await file.read()
    try:
        result = _visual_analyzer.analyze(file.filename, contents)
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logger.exception("CSV analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

    if geo_col:
        counts = df[geo_col].value_counts().head(50)
        points = [{"name": str(k), "count": int(v)} for k, v in counts.items()]
        return {"type": "country", "points": points, "geo_col": geo_col}

    return None
