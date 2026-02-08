"""Ultra backend - Compatibility shim"""
from src.common import ColorFilter, ColorFilterConfig
from src.features.background_removal.ultra import UltraBackend

__all__ = ["ColorFilter", "ColorFilterConfig", "UltraBackend"]
