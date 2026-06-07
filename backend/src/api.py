# from __future__ import annotations
import time
import logging
import os
from contextlib import asynccontextmanager
from typing import Any # Added this import
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Body
from fastapi.exceptions import RequestValidationError # Modified this line to add Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from uuid import uuid4
from src.models import AgentRequest, AgentResponse
from src.grok_client import GrokClient
from src.planner import GrokPlanner
from src.tool_router import ToolRouter
from src.memory import FAISSMemoryStore
from src.engine import ExecutionEngine
import src.tools.summarize as summarize_module
from src.tools.visual_analyzer import VisualAnalyzer

# Rebuild models for Pydantic v2 safety
AgentRequest.model_rebuild()
AgentResponse.model_rebuild()

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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.post("/run-agent", response_model=AgentResponse)
@limiter.limit("10/minute")
async def run_agent(request: Request, agent_request: AgentRequest = Body(...)) -> AgentResponse:
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
            query=agent_request.query,
            session_id=agent_request.session_id,
            max_tasks=agent_request.max_tasks,
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
    """Clear all stored memory AND physical files for a session."""
    import shutil
    from pathlib import Path
    
    # 1. Clear Vector Memory in-memory cache and disk
    if _memory:
        _memory.clear(session_id)
        
    # 2. Clear Physical Data Files
    data_dir = Path(os.getenv("ALLOWED_FILE_DIRS", "./data")) / session_id
    if data_dir.exists():
        shutil.rmtree(data_dir)
        
    return {"session_id": session_id, "cleared": True}


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), session_id: str | None = Form(None)):
    """
    Upload a file (PDF, TXT, MD, CSV, JSON) to a session-specific data directory.
    """
    allowed_extensions = {".pdf", ".txt", ".md", ".csv", ".json", ".xlsx", ".xls"}
    suffix = os.path.splitext(file.filename or "")[-1].lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed_extensions)}"
        )

    # Use existing session or create a new one
    current_session = session_id or str(uuid4())

    # Session-specific data folder
    base_data_dir = os.getenv("ALLOWED_FILE_DIRS", "./data")
    data_dir = os.path.join(base_data_dir, current_session)
    os.makedirs(data_dir, exist_ok=True)

    save_path = os.path.join(data_dir, file.filename)
    contents = await file.read()

    with open(save_path, "wb") as f:
        f.write(contents)

    logger.info("File uploaded: %s (%d bytes) for session %s", save_path, len(contents), current_session)
    
    # Store the fact that this file was uploaded in the session memory
    if _memory:
        _memory.store(current_session, f"Uploaded file: {file.filename}")

    return {
        "filename": file.filename,
        "saved_path": save_path,
        "session_id": current_session,
        "size_bytes": len(contents),
        "message": f"File saved. Use 'read_file' tool with path: {save_path}"
    }


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...), session_id: str | None = Form(None)):
    """
    Modular CSV analysis using the VisualAnalyzer tool.
    Files are saved in session folders for isolation.
    """
    if not _visual_analyzer:
        raise HTTPException(status_code=503, detail="Visual Analyzer not initialized.")

    # Use existing session or create a new one
    current_session = session_id or str(uuid4())
    
    # Session-specific directory
    base_data_dir = os.getenv("ALLOWED_FILE_DIRS", "./data")
    data_dir = os.path.join(base_data_dir, current_session)
    os.makedirs(data_dir, exist_ok=True)

    contents = await file.read()
    save_path = os.path.join(data_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        # Pass filename and content to analyzer
        result = _visual_analyzer.analyze(file.filename, contents)
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        
        # Track in memory
        if _memory:
            _memory.store(current_session, f"Analyzed CSV dataset: {file.filename}")
            
        # Return result with session_id
        result["session_id"] = current_session
        return result
    except Exception as e:
        logger.exception("CSV analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
