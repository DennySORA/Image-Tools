"""
後端模組

提供各種背景移除後端的實作
"""

from .gemini_watermark import GeminiWatermarkBackend
from .image_splitter import ImageSplitterBackend
from .registry import BackendRegistry
from .ultra import UltraBackend


__all__ = [
    "BackendRegistry",
    "GeminiWatermarkBackend",
    "ImageSplitterBackend",
    "UltraBackend",
]
