"""
核心模組 - 定義介面、資料模型和業務邏輯
"""

from .interfaces import BackendProtocol
from .models import BackendInfo, ModelInfo, ProcessConfig, ProcessResult
from .processor import ImageProcessor


__all__ = [
    "BackendProtocol",
    "ProcessConfig",
    "ProcessResult",
    "BackendInfo",
    "ModelInfo",
    "ImageProcessor",
]
