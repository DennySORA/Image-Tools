"""
後端模組

提供各種背景移除後端的實作
"""

from .gemini_watermark import GeminiWatermarkBackend
from .image_splitter import ImageSplitterBackend
from .registry import BackendRegistry


# Note: UltraBackend is imported from src.features.background_removal.ultra
# to avoid circular imports. Import it directly from there if needed.

__all__ = [
    "BackendRegistry",
    "GeminiWatermarkBackend",
    "ImageSplitterBackend",
]
