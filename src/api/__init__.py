"""The HTTP service (task 1e): ``POST /diagnose``, ``GET /conditions``, ``GET /health``."""

from src.api.app import create_app

__all__ = ["create_app"]
