"""Pandas-based data analysis tool for CSV files."""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def data_analysis_tool(file_path: str) -> str:
    """Load a CSV file and return a summary. Searches in ./data if needed."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    path = Path(file_path)
    if not path.exists():
        # Fallback to the configured data directory
        data_dir = Path(os.getenv("ALLOWED_FILE_DIRS", "./data"))
        path = data_dir / path.name

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
