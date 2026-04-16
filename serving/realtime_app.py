"""Compatibility shim for older imports only.

This module is not the canonical serving implementation and is not used by the
Docker or local runtime entrypoints. The real FastAPI application lives at
`serving.app:app`.
"""

from serving.app import app
