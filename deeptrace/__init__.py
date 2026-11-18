"""DeepTrace - multi-signal face-swap forensics."""

from .config import APP_NAME, APP_TAGLINE, APP_VERSION  # noqa: F401
from .engine import analyze_media  # noqa: F401

__all__ = ["analyze_media", "APP_NAME", "APP_TAGLINE", "APP_VERSION"]
