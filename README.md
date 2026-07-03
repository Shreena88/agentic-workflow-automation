# ⚡ Agentic Workflow Automation

An advanced, production-grade AI system using **Grok LLM**, **LangGraph**, and **FAISS Semantic Memory** to automate complex, multi-step workflows.

---

## 🏛️ System Architecture

![Workflow Architecture]

The architecture follows a high-performance, asynchronous pipeline built for scalability and data intelligence.

---

## 🖥️ Dashboard User Interface

The application features a dark-themed, glassmorphic dashboard with two primary workspaces:

### 1. Agent Console
The Agent Console offers a workspace to perform document QA, run web search objectives, and inspect step-by-step agent traces.

![Agent Console Dashboard](./images/image.png)

*   **Context Dropzone**: Index text files, PDFs, or JSON data dynamically into the active session memory.
*   **Memory State & Clear**: Query index counts or reset the FAISS memory cache.
*   **Task Pipeline**: Explains tool steps, runtime status, and inputs/outputs.

### 2. CSV & Spreadsheet Profiler
The CSV Analytics tab automatically parses and profiles tabular data formats.

![CSV Analytics Suite](./images/image%20copy.png)

*   **Dataset Overview**: Instantly count rows, columns, and categorise column data types.
*   **Missing Value Grid**: Inspect cleanliness and missing values counts per field.
*   **AI Interpretation & Charts**: Generate statistical correlations, Line/Bar charts, and map plots coupled with an AI synthesized report.

---

## 📂 Project Directory & File Mapping

| Category | File | Description |
| :--- | :--- | :--- |
| **Core** | `backend/main.py` | Entry point. Initializes environment and boots the Uvicorn server. |
| **API** | `backend/src/api.py` | REST endpoints (`/run-agent`, `/analyze-csv`). Handles singleton lifecycles. |
| **Engine** | `backend/src/engine.py` | LangGraph `StateGraph` logic, nodes, edges, and topological execution. |
| **Memory** | `backend/src/memory.py` | FAISS vector store. Manages per-session indexing and semantic search. |
| **Intelligence** | `backend/src/grok_client.py` | Groq LLM wrapper with token-safe truncation and response synthesis. |
| **Planning** | `backend/src/planner.py` | Converts natural language input into validated, structured task plans. |
| **Tools** | `backend/src/tools/` | Specialized modules for Web Search, PDF/TXT Reading, and Data Analysis. |
| **UI** | `frontend/` | Premium dashboard with Glassmorphism and real-time task tracing. |

---

## 🔥 Key Intelligence Features

### 1. **Token Truncation Safety**
The system implements a multi-layer **Payload Safety** mechanism to prevent `Groq 413` errors. This ensures that even massive search results or documents are condensed *before* being sent for final synthesis, staying within TPM/RPM limits.

### 2. **Proactive Memory Injection**
When files are uploaded, their metadata is instantly embedded into the FAISS memory. This allows the Agent to "remember" your files dynamically—you can simply say *"Summarize that file I just gave you"* and it will succeed without needing the filename.

### 3. **Path-Safe Execution**
The `file_reader` and `visual_analyzer` tools use proactive path resolution. If the agent provides a filename, the tools automatically check the secure `./data/` directory, preventing "File Not Found" errors common in standard agent implementations.

---

## 📊 Visual Intelligence (Visual Analyzer)
The system features a dedicated **Visual Analysis Engine** that goes beyond simple text parsing:
*   **Statistical Profiling**: Rows, columns, mean, std, and missing value detection.
*   **Correlation & Distributions**: Automatically builds configuration for Pie, Bar, Scatter, and Line charts.
*   **Geo-Intelligence**: Dynamically detects Latitude, Longitude, or Country names to build interactive Leaflet maps.

---

## 🚀 How to Run

1.  **Initialize Environment**:
    Navigate to the backend directory and install dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
2.  **Set Credentials**: Update `backend/.env` with your `GROK_API_KEY`.
3.  **Start Backend**:
    Run Uvicorn from the `backend/` folder:
    ```bash
    cd backend
    uvicorn main:app --reload
    ```
4.  **Launch Dashboard**:
    Start the Vite development server in the `frontend/` folder:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

---

