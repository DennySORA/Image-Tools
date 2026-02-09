"""
閾值優化模組

智能選擇最佳 Alpha 閾值，最大化物件偵測品質
"""

from typing import Final

from .detector import DetectedObject


# 常數定義
AUTO_ALPHA_THRESHOLDS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64)
MIN_OBJECTS_FOR_OUTLIER_CHECK: Final[int] = 2
OUTLIER_RATIO_MERGED_THRESHOLD: Final[float] = 1.6
SINGLE_OBJECT_BBOX_AREA_RATIO_THRESHOLD: Final[float] = 0.30
SINGLE_OBJECT_FILL_RATIO_THRESHOLD: Final[float] = 0.70
MIN_STABLE_RUN_LEN: Final[int] = 2
UNSTABLE_SPLIT_PENALTY: Final[float] = 40.0


def should_search_threshold(objects: list[DetectedObject]) -> bool:
    """
    是否應該搜尋更好的閾值

    Args:
        objects: 偵測到的物件列表

    Returns:
        是否需要搜尋
    """
    if len(objects) < MIN_OBJECTS_FOR_OUTLIER_CHECK:
        return False
    return outlier_ratio(objects) >= OUTLIER_RATIO_MERGED_THRESHOLD


def single_object_looks_merged(
    image_width: int,
    image_height: int,
    objects: list[DetectedObject],
) -> bool:
    """
    單個物件是否看起來像合併的

    Args:
        image_width: 圖片寬度
        image_height: 圖片高度
        objects: 偵測到的物件列表

    Returns:
        是否看起來像合併的
    """
    if len(objects) != 1:
        return False
    image_area = max(1, image_width * image_height)
    bbox_area = max(1, objects[0].bbox.area())
    bbox_ratio = bbox_area / image_area
    fill_ratio = objects[0].area_px / bbox_area
    return (
        bbox_ratio >= SINGLE_OBJECT_BBOX_AREA_RATIO_THRESHOLD
        and fill_ratio <= SINGLE_OBJECT_FILL_RATIO_THRESHOLD
    )


def score_candidate(
    threshold: int,
    objects: list[DetectedObject],
    run_len: int,
) -> float:
    """
    評分候選結果

    Args:
        threshold: Alpha 閾值
        objects: 偵測到的物件列表
        run_len: 穩定長度

    Returns:
        分數（越高越好）
    """
    n = len(objects)
    if n <= 0:
        return float("-inf")

    outliers = outlier_ratio(objects)
    small_cnt = small_area_count(objects)
    unstable_penalty = (
        UNSTABLE_SPLIT_PENALTY
        if n >= MIN_OBJECTS_FOR_OUTLIER_CHECK and run_len < MIN_STABLE_RUN_LEN
        else 0.0
    )

    return (
        (n * 100.0)
        + (run_len * 5.0)
        - (outliers * 30.0)
        - (small_cnt * 20.0)
        - unstable_penalty
        - (threshold * 0.1)
    )


def outlier_ratio(objects: list[DetectedObject]) -> float:
    """
    計算離群值比例

    Args:
        objects: 偵測到的物件列表

    Returns:
        最大比例值
    """
    if len(objects) < MIN_OBJECTS_FOR_OUTLIER_CHECK:
        return 1.0

    widths = [o.bbox.width() for o in objects]
    heights = [o.bbox.height() for o in objects]
    bbox_areas = [o.bbox.area() for o in objects]

    mw = max(1, _median_int(widths))
    mh = max(1, _median_int(heights))
    ma = max(1, _median_int(bbox_areas))

    ratios: list[float] = []
    for w, h, a in zip(widths, heights, bbox_areas, strict=True):
        ratios.append(w / mw)
        ratios.append(h / mh)
        ratios.append(a / ma)

    return max(ratios, default=1.0)


def small_area_count(objects: list[DetectedObject]) -> int:
    """
    計算小面積物件數量

    Args:
        objects: 偵測到的物件列表

    Returns:
        小面積物件數
    """
    if not objects:
        return 0
    areas = [o.area_px for o in objects]
    med = max(1, _median_int(areas))
    cutoff = med * 0.25
    return sum(1 for a in areas if a < cutoff)


def _median_int(values: list[int]) -> int:
    """計算中位數"""
    if not values:
        return 0
    values = sorted(values)
    return values[len(values) // 2]
