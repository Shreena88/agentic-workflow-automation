"""Pandas-based data analysis tool for CSV files."""
from __future__ import annotations
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def data_analysis_tool(file_path: str, session_id: str | None = None) -> str:
    """Load a CSV file and return a summary. Searches in session-specific dir if needed."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    path = Path(file_path)
    base_data_dir = Path(os.getenv("ALLOWED_FILE_DIRS", "./data")).resolve()

    if not path.exists():
        if session_id:
            session_path = base_data_dir / session_id / path.name
            if session_path.exists():
                path = session_path
            else:
                path = base_data_dir / path.name
        else:
            path = base_data_dir / path.name

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    if resolved.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {resolved.suffix}")

    df = pd.read_csv(resolved)

    lines = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"\nColumns: {list(df.columns)}",
        f"\nData Types:\n{df.dtypes.to_string()}",
        f"\nStatistical Summary:\n{df.describe(include='all').to_string()}",
        f"\nMissing Values:\n{df.isnull().sum().to_string()}",
    ]
    return "\n".join(lines)
