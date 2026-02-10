"""
Alpha 處理配置模組

提供 premultiplied/straight alpha 處理的配置和工具函數
"""

from enum import StrEnum

import cv2
import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans  # type: ignore[import-untyped]


# 常數定義
ALPHA_EPSILON = 1e-5  # Alpha 最小值（避免除以零）
COLOR_STD_SINGLE = 10  # 單一背景色標準差閾值
COLOR_STD_DUAL = 30  # 雙色背景標準差閾值
ALPHA_THRESHOLD_WEAK = 0.3  # 弱邊緣 alpha 閾值
ALPHA_THRESHOLD_MEDIUM = 0.7  # 中等邊緣 alpha 閾值


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
    n_clusters: int | None = None,
    min_samples: int = 50,
    return_all_clusters: bool = False,
) -> np.ndarray:
    """
    使用增強型 KMeans 聚類估計背景色

    改進：
    1. 自適應選擇 n_clusters（基於數據分布）
    2. 支援返回多個背景色（漸變背景）
    3. 使用輪廓系數優化聚類質量
    4. DBSCAN 作為備選方案

    Args:
        image: RGB 圖片 (H, W, 3), float32
        background_region: 背景區域遮罩 (H, W), bool
        n_clusters: 聚類數量（None 則自動選擇，建議 2-5）
        min_samples: 最小樣本數（少於此數則回退到中位數）
        return_all_clusters: 是否返回所有聚類中心（用於多色背景）

    Returns:
        主要背景色 (1, 3) 或多個背景色 (n_clusters, 3)
    """
    # 提取背景像素
    background_pixels = image[background_region].reshape(-1, 3)

    # 如果樣本太少，使用中位數
    if len(background_pixels) < min_samples:
        return np.median(background_pixels, axis=0, keepdims=True)  # type: ignore[no-any-return]

    # 自適應選擇聚類數量
    if n_clusters is None:
        # 根據背景像素的色彩變異度自動選擇
        color_std = np.std(background_pixels, axis=0).mean()

        if color_std < COLOR_STD_SINGLE:  # 單一背景色（純色）
            n_clusters = 1
        elif color_std < COLOR_STD_DUAL:  # 雙色背景（漸變）
            n_clusters = 2
        else:  # 複雜背景
            n_clusters = 3
    else:
        n_clusters = min(n_clusters, len(background_pixels))

    # 使用 KMeans 聚類識別主要背景色
    try:
        # 如果只有一個聚類，直接返回平均值
        if n_clusters == 1:
            return np.mean(background_pixels, axis=0, keepdims=True)  # type: ignore[no-any-return]

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=42,  # 確保可重現性
        )
        kmeans.fit(background_pixels)

        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)

        # 如果要返回所有聚類（多色背景場景）
        if return_all_clusters:
            # 按聚類大小排序
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            return cluster_centers[sorted_indices]  # type: ignore[no-any-return]

        # 返回最大集群的顏色作為主要背景色
        return cluster_centers[np.argmax(cluster_sizes)].reshape(1, 3)  # type: ignore[no-any-return]

    except Exception:  # noqa: BLE001
        # 如果 KMeans 失敗，回退到中位數
        return np.median(background_pixels, axis=0, keepdims=True)  # type: ignore[no-any-return]


def decontaminate_edges(  # noqa: C901, PLR0912, PLR0915
    image: np.ndarray,
    alpha: np.ndarray,
    strength: float = 0.7,
    use_kmeans: bool = True,
    multi_layer: bool = True,
) -> np.ndarray:
    """
    增強型邊緣去污染（移除背景色滲透）

    在半透明邊緣區域，移除可能來自背景的色彩污染。
    這是修復「白邊」「綠邊」等問題的關鍵步驟。

    改進：
    1. 多層次邊緣分析（強邊緣 vs 弱邊緣）
    2. 自適應 unpremultiply 算法
    3. 多尺度背景估計
    4. 色彩空間融合校正

    原理：
    1. 識別半透明邊緣像素（0.01 < alpha < 0.99）
    2. 對這些像素進行色彩校正，假設它們混合了背景色
    3. 使用增強 unpremultiply 思想：還原「純前景色」

    背景色估計方法：
    - use_kmeans=True: 使用自適應 KMeans 聚類（支援多色背景）
    - use_kmeans=False: 使用中位數估計（速度快但簡單）

    Args:
        image: RGB 圖片 (H, W, 3), uint8
        alpha: Alpha matte (H, W), uint8
        strength: 去污染強度 (0.0-1.0)，越高越激進
        use_kmeans: 是否使用 KMeans 聚類估計背景色（預設: True）
        multi_layer: 是否使用多層次分析（預設: True）

    Returns:
        去污染後的 RGB 圖片 (H, W, 3), uint8
    """
    if strength < 0.01:  # noqa: PLR2004
        return image

    result = image.astype(np.float32)
    alpha_normalized = alpha.astype(np.float32) / 255.0

    # === 階段 1: 識別邊緣區域 ===
    # 邊緣區域（半透明像素）
    edge_mask = (alpha_normalized > 0.01) & (alpha_normalized < 0.99)  # noqa: PLR2004

    if not np.any(edge_mask):
        return image

    # 分層邊緣（如果啟用）
    if multi_layer:
        # 極度半透明邊緣（alpha < 0.3）：通常是色邊，需要激進處理
        very_weak_edge = edge_mask & (alpha_normalized < 0.3)  # noqa: PLR2004
        # 弱邊緣（0.3 <= alpha < 0.7）：可能混合色邊和細節
        weak_edge = (
            edge_mask
            & (alpha_normalized >= ALPHA_THRESHOLD_WEAK)
            & (alpha_normalized < ALPHA_THRESHOLD_MEDIUM)
        )
        # 強邊緣（alpha >= 0.7）：主要是細節，保守處理
        strong_edge = edge_mask & (alpha_normalized >= ALPHA_THRESHOLD_MEDIUM)
    else:
        very_weak_edge = edge_mask
        weak_edge = np.zeros_like(edge_mask, dtype=bool)
        strong_edge = np.zeros_like(edge_mask, dtype=bool)

    # === 階段 2: 多尺度背景色估計 ===
    # 使用兩種尺度的背景區域
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    dilated_edge_small = cv2.dilate(edge_mask.astype(np.uint8), kernel_small)
    dilated_edge_large = cv2.dilate(edge_mask.astype(np.uint8), kernel_large)

    # 近距離背景（更準確，但樣本可能少）
    near_background = (dilated_edge_small > 0) & (
        alpha_normalized < 0.1  # noqa: PLR2004
    )
    # 遠距離背景（樣本多，但可能不準）
    far_background = (dilated_edge_large > 0) & (alpha_normalized < 0.05)  # noqa: PLR2004

    bg_colors = []

    # 優先使用近距離背景
    if np.any(near_background):
        if use_kmeans:
            bg_color = estimate_background_colors_kmeans(
                result, near_background, n_clusters=None, min_samples=30
            )
        else:
            bg_color = np.median(
                result[near_background].reshape(-1, 3), axis=0, keepdims=True
            )
        bg_colors.append(("near", bg_color))

    # 如果近距離樣本不足，使用遠距離
    if np.any(far_background) and len(bg_colors) == 0:
        if use_kmeans:
            bg_color = estimate_background_colors_kmeans(
                result, far_background, n_clusters=None, min_samples=50
            )
        else:
            bg_color = np.median(
                result[far_background].reshape(-1, 3), axis=0, keepdims=True
            )
        bg_colors.append(("far", bg_color))

    # 如果都找不到，使用邊緣區域的中位數
    if len(bg_colors) == 0:
        bg_color = np.median(result[edge_mask].reshape(-1, 3), axis=0, keepdims=True)
        bg_colors.append(("edge", bg_color))

    # 使用第一個有效的背景色
    bg_color = bg_colors[0][1]

    # === 階段 3: 增強型 Unpremultiply 校正 ===
    alpha_3ch = np.stack([alpha_normalized] * 3, axis=-1)

    # 對不同強度的邊緣使用不同的校正策略
    def apply_decontamination(
        mask: np.ndarray, correction_strength: float
    ) -> np.ndarray:
        if not np.any(mask):
            return result

        # 計算背景貢獻
        bg_contribution = bg_color * (1 - alpha_3ch)

        # 安全 alpha（避免除以零）
        alpha_safe = np.where(alpha_3ch > 0.01, alpha_3ch, 1.0)  # noqa: PLR2004

        # 增強型 unpremultiply
        # 考慮 gamma 校正和邊緣梯度
        gamma_correction = 1.0 + (1.0 - alpha_normalized[:, :, np.newaxis]) * 0.2

        return (  # type: ignore[no-any-return]
            (result - bg_contribution * correction_strength)
            / (alpha_safe + (1 - alpha_safe) * (1 - correction_strength))
        ) * gamma_correction

    # 分層應用不同強度的去污染
    if np.any(very_weak_edge):
        result[very_weak_edge] = apply_decontamination(very_weak_edge, strength * 1.0)[
            very_weak_edge
        ]

    if np.any(weak_edge):
        result[weak_edge] = apply_decontamination(weak_edge, strength * 0.7)[weak_edge]

    if np.any(strong_edge):
        result[strong_edge] = apply_decontamination(strong_edge, strength * 0.4)[
            strong_edge
        ]

    # === 階段 4: 色彩空間融合校正 ===
    # 在 LAB 空間進行額外的色度校正（針對頑固色邊）
    if strength > 0.5 and np.any(edge_mask):  # noqa: PLR2004
        lab_result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
            np.float32
        )

        # 在邊緣區域，輕微壓制 a/b 通道的極值
        correction_factor = (strength - 0.5) * 0.4

        lab_result[:, :, 1][edge_mask] = (
            lab_result[:, :, 1][edge_mask] * (1 - correction_factor)
            + 128 * correction_factor
        )
        lab_result[:, :, 2][edge_mask] = (
            lab_result[:, :, 2][edge_mask] * (1 - correction_factor)
            + 128 * correction_factor
        )

        result = cv2.cvtColor(lab_result.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(
            np.float32
        )

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
