"""Unified backend - Compatibility shim"""

from src.common import ColorFilter, ColorFilterConfig
from src.features.background_removal.unified import UnifiedBackend


__all__ = ["ColorFilter", "ColorFilterConfig", "UnifiedBackend"]
