"""
去色邊處理模組

提供多色彩空間分析的進階去色邊功能
"""

import cv2
import numpy as np


# 常數定義
PIXEL_MAX_VALUE = 255
EDGE_ALPHA_MIN = 0.01
EDGE_ALPHA_MAX = 0.99
GRADIENT_THRESHOLD_HIGH = 0.1  # 高梯度閾值（強邊緣）
ALPHA_THRESHOLD_VERY_TRANSPARENT = 0.3  # 極度半透明閾值
ALPHA_THRESHOLD_MEDIUM_TRANSPARENT = 0.6  # 中度半透明閾值
ALPHA_THRESHOLD_CERTAIN_FOREGROUND = 0.95  # 確定前景閾值


def apply_advanced_defringing(  # noqa: PLR0915
    image: np.ndarray, alpha: np.ndarray, strength: float
) -> np.ndarray:
    """
    增強型進階去色邊（多色彩空間分析）

    使用 RGB + LAB + HSV + YCrCb 多重分析來徹底移除色彩污染
    改進：
    1. 添加 YCrCb 色彩空間（更好的色度分離）
    2. 邊緣分類（強邊緣 vs 弱邊緣）
    3. 局部自適應校正
    4. 基於梯度的色彩保留

    Args:
        image: RGB 圖片
        alpha: Alpha matte
        strength: 處理強度 (0.1-1.0)

    Returns:
        處理後的 RGB 圖片
    """
    # 只在中高強度時啟用
    if strength < 0.4:  # noqa: PLR2004
        return image

    result = image.astype(np.float32)
    alpha_normalized = alpha.astype(np.float32) / PIXEL_MAX_VALUE

    # === 階段 1: 智能邊緣分類 ===
    # 計算 alpha 梯度
    grad_x = cv2.Sobel(alpha_normalized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha_normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 邊緣區域分類
    edge_mask = (alpha_normalized > EDGE_ALPHA_MIN) & (
        alpha_normalized < EDGE_ALPHA_MAX
    )
    strong_edge_mask = edge_mask & (
        gradient_magnitude > GRADIENT_THRESHOLD_HIGH
    )  # 強邊緣（頭髮等）
    weak_edge_mask = edge_mask & (
        gradient_magnitude <= GRADIENT_THRESHOLD_HIGH
    )  # 弱邊緣（色邊）

    if not np.any(edge_mask):
        return image

    defringe_strength = (strength - 0.4) * 1.5  # 0->0.9

    # === 方法 1: RGB 色彩平衡（全邊緣） ===
    result = _apply_rgb_balance(
        result, weak_edge_mask, strong_edge_mask, defringe_strength
    )

    # === 方法 2: LAB 色彩空間分析（移除亮度不匹配） ===
    result = _apply_lab_correction(result, edge_mask, defringe_strength)

    # === 方法 3: YCrCb 色度分離（新增，針對色邊特別有效） ===
    result = _apply_ycrcb_correction(result, weak_edge_mask, defringe_strength)

    # === 方法 4: 基於 alpha 的分層處理 ===
    result = _apply_alpha_layered_processing(result, edge_mask, alpha_normalized)

    # === 方法 5: 局部自適應校正（基於鄰域） ===
    result = _apply_foreground_guided_correction(
        result, edge_mask, alpha_normalized, defringe_strength
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_rgb_balance(
    result: np.ndarray,
    weak_edge_mask: np.ndarray,
    strong_edge_mask: np.ndarray,
    defringe_strength: float,
) -> np.ndarray:
    """RGB 色彩平衡校正"""
    r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
    color_mean = (r + g + b) / 3

    # 計算各通道偏離平均值的程度
    r_diff = r - color_mean
    g_diff = g - color_mean
    b_diff = b - color_mean

    # 對弱邊緣更激進，強邊緣保留細節
    weak_correction = defringe_strength * 0.8
    strong_correction = defringe_strength * 0.3

    result[:, :, 0][weak_edge_mask] -= r_diff[weak_edge_mask] * weak_correction
    result[:, :, 1][weak_edge_mask] -= g_diff[weak_edge_mask] * weak_correction
    result[:, :, 2][weak_edge_mask] -= b_diff[weak_edge_mask] * weak_correction

    result[:, :, 0][strong_edge_mask] -= r_diff[strong_edge_mask] * strong_correction
    result[:, :, 1][strong_edge_mask] -= g_diff[strong_edge_mask] * strong_correction
    result[:, :, 2][strong_edge_mask] -= b_diff[strong_edge_mask] * strong_correction

    return result


def _apply_lab_correction(
    result: np.ndarray, edge_mask: np.ndarray, defringe_strength: float
) -> np.ndarray:
    """LAB 色彩空間校正"""
    lab_result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )

    # 在邊緣區域，讓 a/b 通道向灰色（128）靠攏
    lab_result[:, :, 1][edge_mask] = (
        lab_result[:, :, 1][edge_mask] * (1 - defringe_strength * 0.3)
        + 128 * defringe_strength * 0.3
    )
    lab_result[:, :, 2][edge_mask] = (
        lab_result[:, :, 2][edge_mask] * (1 - defringe_strength * 0.3)
        + 128 * defringe_strength * 0.3
    )

    return cv2.cvtColor(lab_result.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(
        np.float32
    )


def _apply_ycrcb_correction(
    result: np.ndarray, weak_edge_mask: np.ndarray, defringe_strength: float
) -> np.ndarray:
    """YCrCb 色度分離校正"""
    bgr_result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ycrcb_result = cv2.cvtColor(bgr_result, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    # 對弱邊緣，壓制 Cr/Cb 通道（色度）向中性值（128）
    # 這對移除綠邊、紫邊等色彩污染特別有效
    ycrcb_result[:, :, 1][weak_edge_mask] = (
        ycrcb_result[:, :, 1][weak_edge_mask] * (1 - defringe_strength * 0.4)
        + 128 * defringe_strength * 0.4
    )
    ycrcb_result[:, :, 2][weak_edge_mask] = (
        ycrcb_result[:, :, 2][weak_edge_mask] * (1 - defringe_strength * 0.4)
        + 128 * defringe_strength * 0.4
    )

    bgr_result = cv2.cvtColor(ycrcb_result.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB).astype(np.float32)


def _apply_alpha_layered_processing(
    result: np.ndarray, edge_mask: np.ndarray, alpha_normalized: np.ndarray
) -> np.ndarray:
    """基於 alpha 的分層處理"""
    # 極度半透明（alpha < 0.3）：激進去色
    very_transparent_mask = edge_mask & (
        alpha_normalized < ALPHA_THRESHOLD_VERY_TRANSPARENT
    )
    if np.any(very_transparent_mask):
        gray_value = np.mean(result, axis=2, keepdims=True)
        result[very_transparent_mask] = (
            result[very_transparent_mask] * 0.5
            + gray_value[very_transparent_mask] * 0.5
        )

    # 中度半透明（0.3 <= alpha < 0.6）：溫和去色
    medium_transparent_mask = (
        edge_mask
        & (alpha_normalized >= ALPHA_THRESHOLD_VERY_TRANSPARENT)
        & (alpha_normalized < ALPHA_THRESHOLD_MEDIUM_TRANSPARENT)
    )
    if np.any(medium_transparent_mask):
        gray_value = np.mean(result, axis=2, keepdims=True)
        result[medium_transparent_mask] = (
            result[medium_transparent_mask] * 0.75
            + gray_value[medium_transparent_mask] * 0.25
        )

    return result


def _apply_foreground_guided_correction(
    result: np.ndarray,
    edge_mask: np.ndarray,
    alpha_normalized: np.ndarray,
    defringe_strength: float,
) -> np.ndarray:
    """局部自適應校正（基於鄰域）"""
    if not np.any(edge_mask):
        return result

    # 膨脹邊緣遮罩找到鄰域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel).astype(bool)

    # 確定前景區域
    certain_foreground = (
        alpha_normalized > ALPHA_THRESHOLD_CERTAIN_FOREGROUND
    ) & ~edge_dilated

    if np.any(certain_foreground):
        # 使用確定前景的平均色進行引導
        fg_mean_color = np.mean(result[certain_foreground].reshape(-1, 3), axis=0)

        # 在邊緣區域向前景色微調（保留細節）
        blend_factor = defringe_strength * 0.15
        result[edge_mask] = result[edge_mask] * (1 - blend_factor) + (
            fg_mean_color * blend_factor
        )

    return result
