"""
資料模型模組 (相容性層)

DEPRECATED: 此模組已遷移至 src.data_model
請改用 from src.data_model import ...

保留此檔案以提供向後相容性
"""

from src.data_model import (
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
