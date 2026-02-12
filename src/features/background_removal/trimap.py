"""
Trimap 處理模組

提供增強型 trimap 建立和 alpha 精修功能
"""

import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)

# 常數定義
PIXEL_MAX_VALUE = 255
TRIMAP_UNKNOWN = 128
GRADIENT_THRESHOLD_HIGH = 0.1  # 高梯度閾值（強邊緣）
GRADIENT_THRESHOLD_LOW = 0.05  # 低梯度閾值（確定區域）


def calculate_trimap_kernel_size(image_size: tuple[int, int], base_kernel: int) -> int:
    """
    根據圖片解析度動態計算 trimap 核大小

    Args:
        image_size: 圖片尺寸 (width, height)
        base_kernel: 基礎核大小 (1024x1024 時的建議值)

    Returns:
        動態調整後的核大小
    """
    width, height = image_size
    max_dim = max(width, height)

    # 以 1024 為基準，動態縮放核大小
    scale_factor = max_dim / 1024.0

    # 計算動態核大小，但限制在合理範圍內
    dynamic_kernel = int(base_kernel * scale_factor)

    # 限制在 5-30 之間（避免過小或過大）
    return max(5, min(30, dynamic_kernel))


def create_trimap(
    alpha: np.ndarray, erode_kernel: int, dilate_kernel: int
) -> np.ndarray:
    """
    建立增強型 trimap（三值圖）

    使用多層次分析建立更精確的 trimap：
    1. 基於閾值的基本分割
    2. 梯度分析識別真實邊緣
    3. 多尺度形態學優化
    4. 自適應調整 unknown region

    Args:
        alpha: Alpha matte (0-255)
        erode_kernel: 腐蝕核大小
        dilate_kernel: 膨脹核大小

    Returns:
        Trimap: 0=背景, 128=未知區, 255=前景
    """
    # 階段 1: 自適應閾值二值化（比固定 127 更智能）
    # 使用 Otsu 方法自動選擇最佳閾值
    threshold_value, binary = cv2.threshold(
        alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    logger.debug("Trimap threshold (Otsu): %d", threshold_value)

    # 階段 2: 梯度分析（識別真實邊緣區域）
    # Alpha 梯度顯示過渡區域
    alpha_float = alpha.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(alpha_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 高梯度區域是真實邊緣，應該保留在 unknown region
    high_gradient_mask = (gradient_magnitude > GRADIENT_THRESHOLD_HIGH).astype(
        np.uint8
    ) * 255

    # 階段 3: 多尺度形態學處理
    # 使用兩種尺度的核來保留細節
    kernel_erode_fine = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, erode_kernel // 2), max(3, erode_kernel // 2))
    )
    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel)
    )

    # 細尺度腐蝕（保留更多細節）
    foreground_fine = cv2.erode(binary, kernel_erode_fine, iterations=1)
    # 粗尺度腐蝕（確定前景）
    foreground_coarse = cv2.erode(binary, kernel_erode, iterations=1)

    # 結合兩種尺度：在高梯度區域使用細尺度
    foreground = np.where(high_gradient_mask > 0, foreground_fine, foreground_coarse)

    # 階段 4: 背景區域處理
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)
    )
    background_inv = cv2.dilate(binary, kernel_dilate, iterations=1)
    background = cv2.bitwise_not(background_inv)

    # 階段 5: 組合 trimap with 梯度引導
    trimap = np.full_like(alpha, TRIMAP_UNKNOWN)  # 預設未知

    # 確定背景（低 alpha + 低梯度）
    certain_background = (background == PIXEL_MAX_VALUE) & (
        gradient_magnitude < GRADIENT_THRESHOLD_LOW
    )
    trimap[certain_background] = 0

    # 確定前景（高 alpha + 低梯度）
    certain_foreground = (foreground == PIXEL_MAX_VALUE) & (
        gradient_magnitude < GRADIENT_THRESHOLD_LOW
    )
    trimap[certain_foreground] = PIXEL_MAX_VALUE

    # Unknown region 自動包含所有高梯度區域
    # 這確保真實邊緣都會被精修

    return trimap


def refine_alpha_in_unknown_region(
    image: np.ndarray, alpha: np.ndarray, trimap: np.ndarray, strength: float
) -> np.ndarray:
    """
    Trimap-based refinement（只處理不確定邊界）

    在未知區域做精細化處理，避免全圖模糊

    Args:
        image: RGB 圖片 (numpy array)
        alpha: 初始 alpha matte
        trimap: Trimap (0/128/255)
        strength: 處理強度 (0.1-1.0)

    Returns:
        精煉後的 alpha
    """
    original_alpha = alpha.astype(np.float32) / PIXEL_MAX_VALUE
    refined_alpha = original_alpha.copy()

    # 找出未知區域
    unknown_mask = trimap == TRIMAP_UNKNOWN

    if not np.any(unknown_mask):
        return alpha

    # 在未知區域使用導向濾波（Guided Filter）進行平滑
    # 這比簡單的高斯模糊保留更多邊緣細節
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 只在未知區域做導向濾波
    radius = int(5 + strength * 15)  # 5-20
    eps = 0.01

    refined_in_unknown = cv2.ximgproc.guidedFilter(
        guide=gray,
        src=refined_alpha,
        radius=radius,
        eps=eps,
    )

    # 只替換未知區域
    refined_alpha[unknown_mask] = refined_in_unknown[unknown_mask]

    # 防止背景洩漏：允許的 alpha 增量與原始 alpha 成正比
    # 導向濾波可能將背景側像素的 alpha 向上平滑，
    # 導致背景色（如綠色）滲入半透明邊緣
    # 漸進式約束：alpha=0 → 不可增加，alpha≥0.3 → 最多增加 0.1
    max_increase = np.clip(original_alpha / 0.3, 0.0, 1.0) * 0.1  # noqa: PLR2004
    refined_alpha = np.minimum(refined_alpha, original_alpha + max_increase)

    return (refined_alpha * PIXEL_MAX_VALUE).astype(np.uint8)


def smooth_alpha_edges(alpha: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Alpha 邊緣平滑：減少噪點和碎片化

    使用 guided filter 在邊緣區域平滑 alpha，保留結構邊緣的銳度。
    再用形態學操作清除孤立噪點。

    Args:
        alpha: Alpha matte (uint8)
        image: RGB 圖片（作為 guided filter 的引導）

    Returns:
        平滑後的 alpha
    """
    alpha_f = alpha.astype(np.float32) / PIXEL_MAX_VALUE

    edge = (alpha_f > 0.02) & (alpha_f < 0.98)  # noqa: PLR2004
    if not np.any(edge):
        return alpha

    # Guided filter：用圖片亮度引導，保留真實邊緣
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    smoothed = cv2.ximgproc.guidedFilter(gray, alpha_f, radius=3, eps=0.01)

    # 只替換邊緣像素（不動前景/背景）
    result_f = alpha_f.copy()
    result_f[edge] = smoothed[edge]

    return (result_f * PIXEL_MAX_VALUE).clip(0, PIXEL_MAX_VALUE).astype(np.uint8)
