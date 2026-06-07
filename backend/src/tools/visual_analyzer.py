"""Automated Visual Data Analyzer for CSV/Excel files.
Generates statistical summaries, chart configurations, and LLM-powered insights.
"""
from __future__ import annotations
import io
import json
import logging
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class VisualAnalyzer:
    def __init__(self, grok_client: Optional[Any] = None) -> None:
        self.grok = grok_client

    def analyze(self, filename: str, contents: bytes) -> Dict[str, Any]:
        """
        Perform a full visual and statistical analysis of a data file.
        Returns a dictionary containing 'summary', 'charts', and 'insights'.
        """
        suffix = os.path.splitext(filename)[1].lower() if filename else ""
        
        try:
            if suffix == ".csv":
                df = pd.read_csv(io.BytesIO(contents))
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                raise ValueError(f"Unsupported file extension: {suffix}")
        except Exception as e:
            logger.error("Failed to parse data file %s: %s", filename, e)
            return {"error": str(e)}

        # Clean data
        df = df.dropna(how="all").reset_index(drop=True)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 1. Summary Statistics
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "missing_values": df.isnull().sum().to_dict(),
            "describe": json.loads(df[num_cols].describe().to_json()) if num_cols else {},
        }

        # 2. Chart Configurations (for Chart.js / Leaflet)
        charts = self._build_charts(df, num_cols, cat_cols)

        # 3. LLM Insights
        insights = "LLM Insights not available."
        if self.grok:
            try:
                stats_text = (
                    f"Dataset: {filename}\n"
                    f"Rows: {summary['rows']}, Columns: {summary['columns']}\n"
                    f"Numeric columns: {', '.join(num_cols)}\n"
                    f"Missing values summary: {sum(summary['missing_values'].values())} total\n"
                    f"Key Statistics (Describe):\n{json.dumps(summary['describe'], indent=2)}"
                )
                prompt = (
                    f"You are a master data analyst. Analyze this dataset and provide:\n"
                    f"1. Executive Summary\n"
                    f"2. Key Observations & Trends\n"
                    f"3. Potential Anomalies\n"
                    f"4. Actionable Business Recommendations\n\n"
                    f"Data Profile:\n{stats_text}"
                )
                insights = self.grok.summarize(prompt)
            except Exception as e:
                logger.warning("Failed to generate LLM insights: %s", e)
                insights = f"Insight generation failed: {e}"

        result = {
            "filename": filename,
            "summary": summary,
            "charts": charts,
            "insights": insights
        }
        return self._sanitize(result)

    def _sanitize(self, obj: Any) -> Any:
        """Recursively replace NaN/Inf float values with None for JSON compliance."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj

    def _build_charts(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> List[Dict[str, Any]]:
        charts = []

        # Bar chart: Means
        if num_cols:
            charts.append({
                "id": "bar_means",
                "type": "bar",
                "title": "Averages per Metric",
                "labels": num_cols,
                "datasets": [{
                    "label": "Mean Value",
                    "data": [round(float(df[c].mean()), 4) for c in num_cols],
                    "backgroundColor": "#8b5cf699", # Indigo
                    "borderColor": "#8b5cf6",
                    "borderWidth": 1,
                }]
            })

        # Line chart: First metric trend
        if num_cols:
            col = num_cols[0]
            subset = df[col].dropna().head(100).tolist()
            charts.append({
                "id": "line_trend",
                "type": "line",
                "title": f"Trend Analysis: {col}",
                "labels": list(range(len(subset))),
                "datasets": [{
                    "label": col,
                    "data": [round(float(v), 4) for v in subset],
                    "borderColor": "#3b82f6", # Blue
                    "backgroundColor": "#3b82f622",
                    "fill": True,
                    "tension": 0.3,
                }]
            })

        # Pie chart: Category distribution
        if cat_cols:
            col = cat_cols[0]
            counts = df[col].value_counts().head(10)
            charts.append({
                "id": "pie_dist",
                "type": "pie",
                "title": f"Distribution: {col}",
                "labels": counts.index.tolist(),
                "datasets": [{
                    "data": [int(v) for v in counts.values],
                    "backgroundColor": ["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#ec4899", "#06b6d4"]
                }]
            })

        # Scatter chart: Correlation between first two numeric columns
        if len(num_cols) >= 2:
            x_col, y_col = num_cols[0], num_cols[1]
            pairs = df[[x_col, y_col]].dropna().head(200)
            charts.append({
                "id": "scatter_corr",
                "type": "scatter",
                "title": f"Correlation: {x_col} vs {y_col}",
                "labels": [],
                "datasets": [{
                    "label": f"{x_col} vs {y_col}",
                    "data": [{"x": float(r[x_col]), "y": float(r[y_col])} for _, r in pairs.iterrows()],
                    "backgroundColor": "#8b5cf688",
                }]
            })

        # Geo Map Check
        geo = self._detect_geo(df)
        if geo:
            charts.append({
                "id": "geo_map",
                "type": "geo",
                "title": "Geographic Intelligence Map",
                "labels": [],
                "datasets": [],
                "geo": geo
            })

        return charts

    def _detect_geo(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        cols_lower = {c.lower(): c for c in df.columns}
        
        # Lat/Lon Search
        lat_names = ["lat", "latitude", "y"]
        lon_names = ["lon", "lng", "longitude", "x"]
        
        lat_col = next((cols_lower[k] for k in lat_names if k in cols_lower), None)
        lon_col = next((cols_lower[k] for k in lon_names if k in cols_lower), None)

        if lat_col and lon_col:
            # find a value column for marker size
            num_cols = df.select_dtypes(include="number").columns.tolist()
            val_col = next((c for c in num_cols if c not in [lat_col, lon_col]), None)
            
            pairs = df[[lat_col, lon_col]].dropna().head(300)
            points = []
            for i, (_, r) in enumerate(pairs.iterrows()):
                if -90 <= float(r[lat_col]) <= 90 and -180 <= float(r[lon_col]) <= 180:
                    val = float(df.loc[_, val_col]) if val_col else 1
                    points.append({
                        "lat": float(r[lat_col]),
                        "lon": float(r[lon_col]),
                        "label": "Location",
                        "value": val
                    })
            if points:
                return {"type": "latlon", "points": points}

        # Country/City Search
        geo_keys = ["country", "state", "city", "location"]
        geo_col = next((cols_lower[k] for k in geo_keys if k in cols_lower), None)
        
        if geo_col:
            counts = df[geo_col].value_counts().head(30)
            return {
                "type": "country",
                "geo_col": geo_col,
                "points": [{"name": str(k), "count": int(v)} for k, v in counts.items()]
            }

        return None
