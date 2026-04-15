"""Compatibility shim for older imports.

The actual runtime entrypoint is now `serving.app:app`.
"""

from serving.app import app
