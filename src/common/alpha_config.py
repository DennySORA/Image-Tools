"""
Alpha 處理配置模組

提供 premultiplied/straight alpha 處理的配置和工具函數
"""

from enum import StrEnum

import cv2
import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans


# 常數定義
ALPHA_EPSILON = 1e-5  # Alpha 最小值（避免除以零）


class AlphaMode(StrEnum):
    """Alpha 通道模式"""

    STRAIGHT = "straight"  # 非預乘 alpha（標準格式，適合大多數場景）
    PREMULTIPLIED = "premultiplied"  # 預乘 alpha（適合特定合成需求）


class ResolutionMode(StrEnum):
    """解析度處理模式"""

    FIXED_1024 = "1024"  # 固定 1024x1024（平衡速度與品質）
    FIXED_1536 = "1536"  # 固定 1536x1536（高品質）
    FIXED_2048 = "2048"  # 固定 2048x2048（極致品質，需要更多顯存）
    ADAPTIVE = "adaptive"  # 自適應（根據輸入自動選擇，最多 2048）


class AlphaConfig(BaseModel):
    """
    Alpha 處理設定

    Attributes:
        mode: Alpha 模式（straight 或 premultiplied）
        edge_decontamination: 是否啟用邊緣去污染
        decontamination_strength: 去污染強度 (0.0-1.0)
    """

    mode: AlphaMode = AlphaMode.STRAIGHT
    edge_decontamination: bool = True
    decontamination_strength: float = Field(default=0.7, ge=0.0, le=1.0)


class ResolutionConfig(BaseModel):
    """
    解析度處理設定

    Attributes:
        mode: 解析度模式
        max_size: 最大尺寸（當模式為 adaptive 時使用）
        maintain_aspect_ratio: 是否保持寬高比
    """

    mode: ResolutionMode = ResolutionMode.FIXED_1024
    max_size: int = Field(default=2048, ge=512, le=4096)
    maintain_aspect_ratio: bool = True


def premultiply_alpha(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    將 straight alpha 轉換為 premultiplied alpha

    Premultiplied alpha: RGB 值已經乘以 alpha 值
    公式：RGB_premul = RGB * alpha

    Args:
        image: RGB 圖片 (H, W, 3), uint8
        alpha: Alpha matte (H, W), uint8

    Returns:
        Premultiplied RGB 圖片 (H, W, 3), uint8
    """
    alpha_normalized = alpha.astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha_normalized] * 3, axis=-1)

    premul = image.astype(np.float32) * alpha_3ch
    return np.clip(premul, 0, 255).astype(np.uint8)


def unpremultiply_alpha(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    將 premultiplied alpha 轉換為 straight alpha

    公式：RGB_straight = RGB_premul / alpha (當 alpha > 0 時)

    Args:
        image: Premultiplied RGB 圖片 (H, W, 3), uint8
        alpha: Alpha matte (H, W), uint8

    Returns:
        Straight alpha RGB 圖片 (H, W, 3), uint8
    """
    alpha_normalized = alpha.astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha_normalized] * 3, axis=-1)

    # 避免除以零
    alpha_safe = np.where(alpha_3ch > ALPHA_EPSILON, alpha_3ch, 1.0)

    unpremul = image.astype(np.float32) / alpha_safe
    return np.clip(unpremul, 0, 255).astype(np.uint8)


def estimate_background_colors_kmeans(
    image: np.ndarray,
    background_region: np.ndarray,
    n_clusters: int = 3,
    min_samples: int = 50,
) -> np.ndarray:
    """
    使用 KMeans 聚類估計背景色

    Args:
        image: RGB 圖片 (H, W, 3), float32
        background_region: 背景區域遮罩 (H, W), bool
        n_clusters: 聚類數量（支援多色背景）
        min_samples: 最小樣本數（少於此數則回退到中位數）

    Returns:
        主要背景色 (1, 3) 或多個背景色 (n_clusters, 3)
    """
    # 提取背景像素
    background_pixels = image[background_region].reshape(-1, 3)

    # 如果樣本太少，使用中位數
    if len(background_pixels) < min_samples:
        return np.median(background_pixels, axis=0, keepdims=True)

    # 使用 KMeans 聚類識別主要背景色
    try:
        kmeans = KMeans(n_clusters=min(n_clusters, len(background_pixels)), n_init=10)
        kmeans.fit(background_pixels)

        # 返回最大集群的中心（主要背景色）
        # 或者返回所有中心（支援多色背景）
        cluster_centers = kmeans.cluster_centers_

        # 計算每個集群的大小
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)

        # 返回最大集群的顏色作為主要背景色
        return cluster_centers[np.argmax(cluster_sizes)].reshape(1, 3)

    except Exception:  # noqa: BLE001
        # 如果 KMeans 失敗，回退到中位數
        return np.median(background_pixels, axis=0, keepdims=True)


def decontaminate_edges(
    image: np.ndarray,
    alpha: np.ndarray,
    strength: float = 0.7,
    use_kmeans: bool = True,
) -> np.ndarray:
    """
    邊緣去污染（移除背景色滲透）

    在半透明邊緣區域，移除可能來自背景的色彩污染。
    這是修復「白邊」「綠邊」等問題的關鍵步驟。

    原理：
    1. 識別半透明邊緣像素（0.01 < alpha < 0.99）
    2. 對這些像素進行色彩校正，假設它們混合了背景色
    3. 使用 unpremultiply 思想：還原「純前景色」

    背景色估計方法：
    - use_kmeans=True: 使用 KMeans 聚類識別主要背景色（支援多色背景）
    - use_kmeans=False: 使用中位數估計（速度快但不支援多色）

    Args:
        image: RGB 圖片 (H, W, 3), uint8
        alpha: Alpha matte (H, W), uint8
        strength: 去污染強度 (0.0-1.0)，越高越激進
        use_kmeans: 是否使用 KMeans 聚類估計背景色（預設: True）

    Returns:
        去污染後的 RGB 圖片 (H, W, 3), uint8
    """
    if strength < 0.01:  # noqa: PLR2004
        return image

    result = image.astype(np.float32)
    alpha_normalized = alpha.astype(np.float32) / 255.0

    # 識別邊緣區域（半透明像素）
    edge_mask = (alpha_normalized > 0.01) & (alpha_normalized < 0.99)  # noqa: PLR2004

    if not np.any(edge_mask):
        return image

    # 方法 1: Unpremultiply-like correction
    # 假設邊緣像素 = 前景色 * alpha + 背景色 * (1-alpha)
    # 我們要還原前景色，需要減去背景色的影響

    # 估計背景色（使用邊緣外圍的像素）
    # 擴展邊緣遮罩以獲取背景樣本
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_edge = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=2)
    background_region = (dilated_edge > 0) & (alpha_normalized < 0.1)  # noqa: PLR2004

    if np.any(background_region):
        # 估計背景色
        if use_kmeans:
            # 使用 KMeans 聚類估計背景色（更智能，支援多色背景）
            bg_color = estimate_background_colors_kmeans(
                result, background_region, n_clusters=3, min_samples=50
            )
        else:
            # 使用中位數估計（速度快但簡單）
            bg_color = np.median(
                result[background_region].reshape(-1, 3), axis=0, keepdims=True
            )
    else:
        # 如果找不到背景區域，使用邊緣區域的平均色
        bg_color = np.mean(result[edge_mask].reshape(-1, 3), axis=0, keepdims=True)

    # 在邊緣區域應用去污染
    alpha_3ch = np.stack([alpha_normalized] * 3, axis=-1)

    # 計算「純前景色」估計
    # foreground = (pixel - bg * (1-alpha)) / alpha
    alpha_safe = np.where(alpha_3ch > 0.01, alpha_3ch, 1.0)  # noqa: PLR2004
    bg_contribution = bg_color * (1 - alpha_3ch)

    foreground_estimate = (result - bg_contribution * strength) / (
        alpha_safe + (1 - alpha_safe) * (1 - strength)
    )

    # 只在邊緣區域替換
    result[edge_mask] = foreground_estimate[edge_mask]

    return np.clip(result, 0, 255).astype(np.uint8)


def calculate_adaptive_resolution(
    original_size: tuple[int, int], max_size: int = 2048
) -> tuple[int, int]:
    """
    計算自適應解析度

    根據原圖大小，選擇合適的推論解析度：
    - 小圖（< 512）：放大到至少 512
    - 中圖（512-1024）：保持原樣或稍微放大
    - 大圖（> 1024）：縮小到 max_size 以內

    Args:
        original_size: 原始圖片尺寸 (width, height)
        max_size: 最大允許尺寸

    Returns:
        推論解析度 (width, height)
    """
    width, height = original_size
    max_dim = max(width, height)

    # 小圖：放大到至少 512
    if max_dim < 512:  # noqa: PLR2004
        scale = 512 / max_dim
        return (int(width * scale), int(height * scale))

    # 中圖：保持在 512-1536 之間
    if max_dim < 1536:  # noqa: PLR2004
        return original_size

    # 大圖：縮小到 max_size
    if max_dim > max_size:
        scale = max_size / max_dim
        return (int(width * scale), int(height * scale))

    return original_size
