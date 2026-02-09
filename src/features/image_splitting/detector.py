"""
物件偵測模組

從 Alpha 通道進行連通分量偵測，識別獨立物件
"""

import logging
from array import array
from typing import Final

from .geometry import BBox
from .union_find import UnionFind, UnionFindDynamic


logger = logging.getLogger(__name__)

# 常數定義
ALPHA_MAX: Final[int] = 255


class DetectedObject:
    """
    檢測到的物件

    Attributes:
        bbox: 邊界框
        area_px: 像素面積
    """

    __slots__ = ("bbox", "area_px")

    def __init__(self, bbox: BBox, area_px: int) -> None:
        self.bbox = bbox
        self.area_px = area_px


def detect_components_from_alpha(
    alpha: bytes,
    *,
    width: int,
    height: int,
    alpha_threshold: int,
) -> list[DetectedObject]:
    """
    從 Alpha 通道檢測連通分量

    Args:
        alpha: Alpha 通道位元組
        width: 圖片寬度
        height: 圖片高度
        alpha_threshold: Alpha 閾值

    Returns:
        檢測到的物件列表
    """
    labels, uf, w2 = _label_alpha_to_buffer(
        alpha, width=width, height=height, alpha_threshold=alpha_threshold
    )
    return _objects_from_labeled_buffer(
        labels, uf, width=width, height=height, width2=w2
    )


def merge_objects(
    objects: list[DetectedObject],
    *,
    merge_pad_px: int,
    max_width: int,
    max_height: int,
) -> list[DetectedObject]:
    """
    合併相近的物件

    Args:
        objects: 物件列表
        merge_pad_px: 合併填充像素
        max_width: 最大寬度
        max_height: 最大高度

    Returns:
        合併後的物件列表
    """
    expanded: list[BBox] = [
        obj.bbox.expand(merge_pad_px, max_width=max_width, max_height=max_height)
        for obj in objects
    ]
    uf = UnionFind(len(objects))

    # 檢查重疊
    for i in range(len(objects)):
        bi = expanded[i]
        for j in range(i + 1, len(objects)):
            if bi.overlaps(expanded[j]):
                uf.union(i, j)

    # 分組
    groups: dict[int, list[int]] = {}
    for i in range(len(objects)):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    # 合併
    merged: list[DetectedObject] = []
    for idxs in groups.values():
        bbox = objects[idxs[0]].bbox
        area = 0
        for i in idxs:
            bbox = bbox.union(objects[i].bbox)
            area += objects[i].area_px
        merged.append(DetectedObject(bbox=bbox, area_px=area))

    return merged


# === 內部函數 ===


def _label_alpha_to_buffer(
    alpha: bytes,
    *,
    width: int,
    height: int,
    alpha_threshold: int,
) -> tuple[array[int], UnionFindDynamic, int]:
    """將 Alpha 通道標記到緩衝區 (8-連通標記)"""
    w: Final[int] = width
    h: Final[int] = height
    w2: Final[int] = w + 2  # 加 1 像素邊框避免邊界檢查
    h2: Final[int] = h + 2

    labels: array[int] = array("I", [0]) * (w2 * h2)
    uf = UnionFindDynamic()

    mv = memoryview(alpha)
    thr: Final[int] = alpha_threshold

    for y in range(h):
        alpha_row = y * w
        label_row = (y + 1) * w2 + 1

        for x in range(w):
            if mv[alpha_row + x] <= thr:
                continue

            idx = label_row + x

            # 檢查 8 個鄰居 (上、左上、左、右上)
            neighbors = (
                labels[idx - 1],  # 左
                labels[idx - w2],  # 上
                labels[idx - w2 - 1],  # 左上
                labels[idx - w2 + 1],  # 右上
            )

            base = _min_nonzero(*neighbors)
            if base == 0:
                # 新分量
                labels[idx] = uf.make_set()
                continue

            labels[idx] = base
            _union_neighbors(uf, base, neighbors)

    return labels, uf, w2


def _objects_from_labeled_buffer(
    labels: array[int],
    uf: UnionFindDynamic,
    *,
    width: int,
    height: int,
    width2: int,
) -> list[DetectedObject]:
    """從標記緩衝區提取物件"""
    stats: dict[int, tuple[int, int, int, int, int]] = {}
    # tuple = (left, top, right_excl, bottom_excl, area)

    for y in range(height):
        label_row = (y + 1) * width2 + 1
        for x in range(width):
            idx = label_row + x
            lab = labels[idx]
            if lab == 0:
                continue

            root = uf.find(lab)
            labels[idx] = root

            entry = stats.get(root)
            if entry is None:
                stats[root] = (x, y, x + 1, y + 1, 1)
                continue

            left, top, right, bottom, area = entry
            stats[root] = (
                min(left, x),
                min(top, y),
                max(right, x + 1),
                max(bottom, y + 1),
                area + 1,
            )

    return [
        DetectedObject(
            bbox=BBox(left=left, top=top, right=right, bottom=bottom),
            area_px=area,
        )
        for left, top, right, bottom, area in stats.values()
    ]


def _min_nonzero(*values: int) -> int:
    """取最小非零值"""
    base = 0
    for v in values:
        if v and (base == 0 or v < base):
            base = v
    return base


def _union_neighbors(
    uf: UnionFindDynamic, base: int, neighbors: tuple[int, int, int, int]
) -> None:
    """合併鄰居"""
    for n in neighbors:
        if n and n != base:
            uf.union(base, n)
