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
    estimate_background_colors_kmeans,
    premultiply_alpha,
    unpremultiply_alpha,
)
from .color_filter import ColorFilter, ColorFilterConfig
from .model_compat import load_pretrained_no_meta
from .preset_config import (
    BackgroundRemovalPreset,
    PresetLevel,
    get_preset,
    list_presets,
    print_preset_comparison,
)


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
    "estimate_background_colors_kmeans",
    "calculate_adaptive_resolution",
    "BackgroundRemovalPreset",
    "PresetLevel",
    "get_preset",
    "list_presets",
    "print_preset_comparison",
    "load_pretrained_no_meta",
]
