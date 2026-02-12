"""
色彩過濾處理模組

提供純色背景（綠幕、白色、黑色）的專門優化
"""

import logging

import cv2
import numpy as np

from src.common import ColorFilter, ColorFilterConfig


logger = logging.getLogger(__name__)

# 常數定義
PIXEL_MAX_VALUE = 255
BLACK_THRESHOLD = 30
WHITE_THRESHOLD = 225


def apply_color_filter(
    image: np.ndarray,
    alpha: np.ndarray,
    color_filter: ColorFilterConfig,
) -> np.ndarray:
    """
    色彩過濾（針對純色背景）

    綠色模式：只修改 RGB（不碰 alpha），避免過度移除。
    黑/白模式：修改 alpha mask。

    Args:
        image: RGB 圖片（會被 in-place 修改）
        alpha: 當前 alpha matte
        color_filter: 色彩過濾配置

    Returns:
        精煉後的 alpha（綠色模式下不變）
    """
    if not color_filter.enabled or color_filter.color == ColorFilter.NONE:
        return alpha

    logger.debug("Applying color filter: %s", color_filter.color.value)

    if color_filter.color == ColorFilter.GREEN:
        # 綠色模式：純 RGB despill，不修改 alpha
        apply_green_despill(image, alpha, color_filter.edge_refine_strength)
        return alpha

    # 黑/白模式：使用 alpha mask
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    if color_filter.color == ColorFilter.BLACK:
        l_channel = lab[:, :, 0]
        color_mask = (l_channel < BLACK_THRESHOLD).astype(np.uint8) * PIXEL_MAX_VALUE

    elif color_filter.color == ColorFilter.WHITE:
        l_channel = lab[:, :, 0]
        color_mask = (l_channel > WHITE_THRESHOLD).astype(np.uint8) * PIXEL_MAX_VALUE

    else:
        return alpha

    # 優化遮罩
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(  # type: ignore[assignment]
        color_mask, cv2.MORPH_CLOSE, kernel
    )
    color_mask = cv2.morphologyEx(  # type: ignore[assignment]
        color_mask, cv2.MORPH_OPEN, kernel
    )

    # 腐蝕以避免顏色邊
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_mask = cv2.erode(  # type: ignore[assignment]
        color_mask, erode_kernel, iterations=1
    )

    # 羽化
    color_mask = cv2.GaussianBlur(color_mask, (7, 7), 0)  # type: ignore[assignment]

    # 與現有 alpha 合併（取最小值）
    foreground_mask = PIXEL_MAX_VALUE - color_mask
    return np.minimum(alpha, foreground_mask)  # type: ignore[no-any-return]


def apply_green_despill(image: np.ndarray, alpha: np.ndarray, strength: float) -> None:
    """
    綠色 despill：只修改 RGB，不碰 alpha。

    使用「距離邊緣的距離」決定校正強度（而非 alpha 值），
    因為模型可能將邊緣像素標記為完全不透明，但 RGB 仍帶有綠色污染。

    校正邏輯：
    - 距離背景越近 → 校正越強（proximity-based）
    - 半透明像素 → 也按 (1-alpha) 校正
    - 取兩者的最大值，確保不遺漏

    Args:
        image: RGB 圖片（in-place 修改）
        alpha: Alpha matte（不修改）
        strength: 校正強度
    """
    alpha_norm = alpha.astype(np.float32) / PIXEL_MAX_VALUE

    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    # 計算綠色過剩量
    rb_avg = (r + b) / 2
    green_excess = np.maximum(g - rb_avg, 0)

    # 基於距離的校正權重：距離背景越近 → 綠色污染越可能
    bg_mask = (alpha_norm < 0.1).astype(np.uint8)  # noqa: PLR2004
    if np.any(bg_mask):
        fg_mask = (1 - bg_mask).astype(np.uint8)
        distance = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
        # 距離衰減：明亮綠背景的光溢可達 200px+
        max_dist = 200.0
        proximity = np.clip(1.0 - distance / max_dist, 0, 1).astype(np.float32)
    else:
        # 無背景區域時（全前景），退回 alpha 權重
        proximity = np.zeros_like(alpha_norm)

    # 也用 alpha 權重（半透明像素即使離邊緣遠也需校正）
    alpha_weight = 1.0 - alpha_norm

    # 取最大值：確保兩種情況都覆蓋
    effective_weight = np.maximum(proximity, alpha_weight)

    # 只對前景像素校正
    effective_weight[alpha_norm < 0.01] = 0  # noqa: PLR2004

    correction = green_excess * effective_weight * strength

    image[:, :, 1] = np.clip(g - correction, 0, 255).astype(np.uint8)


def suppress_residual_green(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    去污染後的殘留綠色抑制

    使用距離邊緣的距離（而非 alpha 值）來決定處理範圍，
    因為不透明像素也可能帶有綠色污染（模型將綠色污染像素標為不透明）。

    Args:
        image: RGB 圖片
        alpha: Alpha matte

    Returns:
        修正後的 RGB 圖片
    """
    alpha_norm = alpha.astype(np.float32) / PIXEL_MAX_VALUE
    result = image.astype(np.float32)

    r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]

    # 使用距離而非 alpha 來判斷「邊緣附近」
    bg_mask = (alpha_norm < 0.1).astype(np.uint8)  # noqa: PLR2004
    if np.any(bg_mask):
        fg_mask = (1 - bg_mask).astype(np.uint8)
        distance = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
        near_edge = (distance < 100) & (alpha_norm > 0.01)  # noqa: PLR2004
    else:
        # 無背景區域時，退回使用 alpha 範圍
        near_edge = (alpha_norm > 0.01) & (alpha_norm < 0.99)  # noqa: PLR2004
        distance = np.full_like(alpha_norm, 20.0)  # 預設中距離

    if not np.any(near_edge):
        return image

    # 綠色過剩 = G - max(R, B)
    green_excess = g - np.maximum(r, b)

    # 閾值 3：即使微小的綠色過剩也要處理
    green_problem = near_edge & (green_excess > 3)  # noqa: PLR2004
    if not np.any(green_problem):
        return image

    # 容差基於距離：越靠近邊緣越嚴格
    proximity = np.clip(1.0 - distance / 100.0, 0, 1)
    tolerance = (1.0 - proximity) * 5  # 邊緣=0, 100px處=5
    target_g = np.maximum(r, b) + tolerance
    result[:, :, 1][green_problem] = np.minimum(
        g[green_problem], target_g[green_problem]
    )

    return np.clip(result, 0, 255).astype(np.uint8)
