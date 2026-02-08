"""
核心模組 - 定義介面、資料模型和業務邏輯
"""

from src.data_model import BackendInfo, ModelInfo, ProcessConfig, ProcessResult

from .interfaces import BackendProtocol
from .processor import ImageProcessor


__all__ = [
    "BackendProtocol",
    "ProcessConfig",
    "ProcessResult",
    "BackendInfo",
    "ModelInfo",
    "ImageProcessor",
]
