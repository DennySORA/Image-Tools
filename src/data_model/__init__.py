"""
資料模型模組

提供應用程式的核心資料結構，使用 Pydantic 進行驗證
"""

from .core import (
    SUPPORTED_EXTENSIONS,
    BackendInfo,
    ImageFile,
    ModelInfo,
    ProcessConfig,
    ProcessResult,
    is_supported_image,
)

__all__ = [
    "BackendInfo",
    "ImageFile",
    "ModelInfo",
    "ProcessConfig",
    "ProcessResult",
    "SUPPORTED_EXTENSIONS",
    "is_supported_image",
]
