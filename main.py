"""Entry point: run with `uvicorn main:app --reload`"""
# Load .env FIRST before any other imports read os.environ
from dotenv import load_dotenv
load_dotenv(override=True)

import logging
from src.api import app  # noqa: F401 - re-exported for uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
