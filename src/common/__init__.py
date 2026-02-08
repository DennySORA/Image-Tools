"""
共用模組

提供在多個功能間共用的工具和定義
"""

from .alpha_config import (
    AlphaConfig,
    AlphaMode,
    ResolutionConfig,
    ResolutionMode,
    calculate_adaptive_resolution,
    decontaminate_edges,
    premultiply_alpha,
    unpremultiply_alpha,
)
from .color_filter import ColorFilter, ColorFilterConfig


__all__ = [
    "ColorFilter",
    "ColorFilterConfig",
    "AlphaConfig",
    "AlphaMode",
    "ResolutionConfig",
    "ResolutionMode",
    "premultiply_alpha",
    "unpremultiply_alpha",
    "decontaminate_edges",
    "calculate_adaptive_resolution",
]
