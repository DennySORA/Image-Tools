"""
工具模組 (相容性層)

DEPRECATED: 工具已遷移至對應功能模組
- splitter, union_find, geometry -> src.features.image_splitting
"""

from src.features.image_splitting import geometry, splitter, union_find

__all__ = ["geometry", "splitter", "union_find"]
