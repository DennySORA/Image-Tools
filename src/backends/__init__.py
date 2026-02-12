"""
後端模組

提供各種背景移除後端的實作

注意：使用延遲導入避免循環依賴
"""

from typing import TYPE_CHECKING

from .registry import BackendRegistry


if TYPE_CHECKING:
    from .batch_compare import BatchCompareBackend
    from .gemini_watermark import GeminiWatermarkBackend
    from .image_splitter import ImageSplitterBackend
    from .ultra import UltraBackend


def __getattr__(name: str) -> type:
    """延遲導入後端類別，避免循環依賴"""
    if name == "BatchCompareBackend":
        from .batch_compare import BatchCompareBackend

        return BatchCompareBackend
    if name == "GeminiWatermarkBackend":
        from .gemini_watermark import GeminiWatermarkBackend

        return GeminiWatermarkBackend
    if name == "ImageSplitterBackend":
        from .image_splitter import ImageSplitterBackend

        return ImageSplitterBackend
    if name == "UltraBackend":
        from .ultra import UltraBackend

        return UltraBackend
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "BackendRegistry",
    "BatchCompareBackend",
    "GeminiWatermarkBackend",
    "ImageSplitterBackend",
    "UltraBackend",
]
