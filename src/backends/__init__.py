"""
後端模組

提供各種背景移除後端的實作
"""

from .backgroundremover import BackgroundRemoverBackend
from .gemini_watermark import GeminiWatermarkBackend
from .greenscreen import GreenScreenBackend
from .image_splitter import ImageSplitterBackend
from .registry import BackendRegistry
from .rembg import RembgBackend
from .transparent_bg import TransparentBgBackend
from .ultra import UltraBackend
from .unified import UnifiedBackend


__all__ = [
    "BackendRegistry",
    "RembgBackend",
    "TransparentBgBackend",
    "BackgroundRemoverBackend",
    "GreenScreenBackend",
    "GeminiWatermarkBackend",
    "ImageSplitterBackend",
    "UnifiedBackend",
    "UltraBackend",
]
