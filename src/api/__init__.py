"""
API package exports.
"""
from .server import create_fastapi_app, get_detector

__all__ = ["create_fastapi_app", "get_detector"]
