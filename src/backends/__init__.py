"""
後端模組

提供各種背景移除後端的實作
"""

from .batch_compare import BatchCompareBackend
from .gemini_watermark import GeminiWatermarkBackend
from .image_splitter import ImageSplitterBackend
from .registry import BackendRegistry
from .ultra import UltraBackend


__all__ = [
    "BackendRegistry",
    "BatchCompareBackend",
    "GeminiWatermarkBackend",
    "ImageSplitterBackend",
    "UltraBackend",
]
