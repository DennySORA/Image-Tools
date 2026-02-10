"""
極致背景移除後端 - 追求最強效果（非商用）

整合最先進的技術與完整的處理流程：
1. BRIA RMBG-2.0：基於 BiRefNet 的增強版，專業級訓練數據
2. Trimap-based Refinement：只在不確定邊界處理，避免過度模糊
3. Advanced Defringing：多色彩空間分析，徹底移除色彩污染
4. Color Filter：針對純色背景的專門優化

授權注意：
- BRIA RMBG-2.0 使用 CC BY-NC 4.0（非商用）授權
- 僅供個人使用、研究、教育用途
- 商業使用需與 BRIA 另行協議

技術參考：
- RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0
- BiRefNet: https://github.com/ZhengPeng7/BiRefNet
- Alpha Matting: https://github.com/pymatting/pymatting
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms  # type: ignore[import-untyped]

from src.backends.registry import BackendRegistry
from src.common import (
    AlphaConfig,
    ColorFilter,
    ColorFilterConfig,
    ResolutionConfig,
    ResolutionMode,
    calculate_adaptive_resolution,
    decontaminate_edges,
    load_pretrained_no_meta,
    premultiply_alpha,
)
from src.common.preset_config import PresetLevel, get_preset
from src.core.interfaces import BaseBackend


logger = logging.getLogger(__name__)

# 常數定義
PIXEL_MAX_VALUE = 255
TRIMAP_UNKNOWN = 128
BLACK_THRESHOLD = 30
WHITE_THRESHOLD = 225
EDGE_ALPHA_MIN = 0.01
EDGE_ALPHA_MAX = 0.99
ALPHA_CHANNEL_INDEX = 3
GRADIENT_THRESHOLD_HIGH = 0.1  # 高梯度閾值（強邊緣）
GRADIENT_THRESHOLD_LOW = 0.05  # 低梯度閾值（確定區域）
ALPHA_THRESHOLD_VERY_TRANSPARENT = 0.3  # 極度半透明閾值
ALPHA_THRESHOLD_MEDIUM_TRANSPARENT = 0.6  # 中度半透明閾值
ALPHA_THRESHOLD_CERTAIN_FOREGROUND = 0.95  # 確定前景閾值


@BackendRegistry.register("ultra")
class UltraBackend(BaseBackend):
    """
    極致背景移除後端（非商用）

    使用最強的開源技術組合：
    - BRIA RMBG-2.0：專業級 alpha matte 輸出
    - Trimap Refinement：智能邊界處理
    - Multi-space Defringing：徹底去除色邊
    - Adaptive Color Filter：純色背景優化

    ⚠️ 注意：僅供非商用使用（CC BY-NC 4.0）
    """

    name: ClassVar[str] = "ultra"
    description: ClassVar[str] = "極致方案 - RMBG-2.0 + Trimap + Defringing（非商用）"

    # RMBG-2.0 模型配置
    MODEL_NAME: ClassVar[str] = "briaai/RMBG-2.0"
    # INPUT_SIZE 現在動態決定，基於 ResolutionConfig

    # Trimap 參數
    TRIMAP_ERODE_KERNEL: ClassVar[int] = 10  # 腐蝕核大小
    TRIMAP_DILATE_KERNEL: ClassVar[int] = 10  # 膨脹核大小

    def __init__(  # noqa: PLR0913
        self,
        model: str = "auto",  # noqa: ARG002
        strength: float = 0.8,
        color_filter: ColorFilterConfig | None = None,
        use_trimap_refine: bool = True,
        alpha_config: AlphaConfig | None = None,
        resolution_config: ResolutionConfig | None = None,
        device: str | None = None,
    ):
        """
        初始化極致後端

        Args:
            model: 模型名稱（保留參數以符合介面，實際固定使用 RMBG-2.0）
            strength: 處理強度 (0.1-1.0)
                     - 0.1-0.3: 保守（保留更多細節，可能有殘留）
                     - 0.4-0.7: 平衡（推薦）
                     - 0.8-1.0: 激進（最大移除，可能損失細節）
            color_filter: 色彩過濾設定
            use_trimap_refine: 是否使用 trimap refinement（推薦開啟）
            alpha_config: Alpha 處理設定（邊緣去污染、輸出模式）
            resolution_config: 解析度設定（1024/1536/2048/adaptive）
            device: 計算設備（cuda/cpu），None 則自動選擇
        """
        super().__init__(strength=strength)
        self.color_filter = color_filter or ColorFilterConfig()
        self.use_trimap_refine = use_trimap_refine
        self.alpha_config = alpha_config or AlphaConfig()
        self.resolution_config = resolution_config or ResolutionConfig()

        # 設備配置（CUDA → MPS → CPU）
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 模型和轉換器
        self._model: Any = None
        self._transform_cache: dict[tuple[int, int], transforms.Compose] = {}

        logger.info(
            "Ultra backend: device=%s, strength=%.1f, trimap=%s (CC BY-NC 4.0)",
            self.device,
            self.strength,
            self.use_trimap_refine,
        )

    def load_model(self) -> None:
        """載入 RMBG-2.0 模型"""
        logger.info("Loading RMBG-2.0 model...")

        # 載入模型（跳過 meta device 初始化，避免自訂 birefnet.py 呼叫 .item() 失敗）
        self._model = load_pretrained_no_meta(self.MODEL_NAME)
        self._model.to(self.device)
        self._model.eval()

        # 啟用 TF32（Ampere+ GPU 自動加速 float32 矩陣運算）
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        logger.info("RMBG-2.0 model loaded on %s", self.device)

    def _apply_rmbg_segmentation(self, image: Image.Image) -> np.ndarray:
        """
        階段 1: 使用 RMBG-2.0 進行主要分割

        Args:
            image: 輸入圖片 (RGB)

        Returns:
            Alpha matte (0-255 uint8 numpy array)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # 保存原始尺寸
        original_size = image.size

        # 根據解析度配置決定推論尺寸
        inference_size = self._get_inference_size(original_size)

        logger.debug(
            "Resolution: %dx%d → %dx%d (mode: %s)",
            original_size[0],
            original_size[1],
            inference_size[0],
            inference_size[1],
            self.resolution_config.mode.value,
        )

        # 取得或建立轉換器（快取以避免重複建立）
        transform = self._transform_cache.get(inference_size)
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(inference_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self._transform_cache[inference_size] = transform

        # 轉換並推論
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 取最高解析度輸出（[-1]），sigmoid 轉為 0-1 機率，移除 batch/channel 維度
            output = self._model(input_tensor)[-1].sigmoid().squeeze()

        # 轉為 numpy 並調整大小回原圖
        alpha = output.cpu().numpy()
        alpha = (alpha * PIXEL_MAX_VALUE).clip(0, PIXEL_MAX_VALUE).astype(np.uint8)

        # Resize 回原始尺寸（使用高品質插值）
        return cv2.resize(alpha, original_size, interpolation=cv2.INTER_CUBIC)  # type: ignore[no-any-return]

    def _get_inference_size(self, original_size: tuple[int, int]) -> tuple[int, int]:
        """
        根據配置決定推論解析度

        Args:
            original_size: 原始圖片尺寸 (width, height)

        Returns:
            推論解析度 (width, height)
        """
        mode = self.resolution_config.mode

        if mode == ResolutionMode.ADAPTIVE:
            # 自適應模式：根據原圖大小智能選擇
            return calculate_adaptive_resolution(
                original_size, self.resolution_config.max_size
            )

        # 固定解析度模式
        size_map = {
            ResolutionMode.FIXED_1024: 1024,
            ResolutionMode.FIXED_1536: 1536,
            ResolutionMode.FIXED_2048: 2048,
        }

        target_size = size_map.get(mode, 1024)
        return (target_size, target_size)

    def _calculate_trimap_kernel_size(
        self, image_size: tuple[int, int], base_kernel: int
    ) -> int:
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

    def _create_trimap(
        self, alpha: np.ndarray, erode_kernel: int, dilate_kernel: int
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
        foreground = np.where(
            high_gradient_mask > 0, foreground_fine, foreground_coarse
        )

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

    def _refine_alpha_in_unknown_region(
        self, image: np.ndarray, alpha: np.ndarray, trimap: np.ndarray
    ) -> np.ndarray:
        """
        階段 2: Trimap-based refinement（只處理不確定邊界）

        在未知區域做精細化處理，避免全圖模糊

        Args:
            image: RGB 圖片 (numpy array)
            alpha: 初始 alpha matte
            trimap: Trimap (0/128/255)

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
        radius = int(5 + self.strength * 15)  # 5-20
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

    def _apply_advanced_defringing(  # noqa: PLR0915
        self, image: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """
        階段 3: 增強型進階去色邊（多色彩空間分析）

        使用 RGB + LAB + HSV + YCrCb 多重分析來徹底移除色彩污染
        改進：
        1. 添加 YCrCb 色彩空間（更好的色度分離）
        2. 邊緣分類（強邊緣 vs 弱邊緣）
        3. 局部自適應校正
        4. 基於梯度的色彩保留

        Args:
            image: RGB 圖片
            alpha: Alpha matte

        Returns:
            處理後的 RGB 圖片
        """
        # 只在中高強度時啟用
        if self.strength < 0.4:  # noqa: PLR2004
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

        defringe_strength = (self.strength - 0.4) * 1.5  # 0->0.9

        # === 方法 1: RGB 色彩平衡（全邊緣） ===
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

        result[:, :, 0][strong_edge_mask] -= (
            r_diff[strong_edge_mask] * strong_correction
        )
        result[:, :, 1][strong_edge_mask] -= (
            g_diff[strong_edge_mask] * strong_correction
        )
        result[:, :, 2][strong_edge_mask] -= (
            b_diff[strong_edge_mask] * strong_correction
        )

        # === 方法 2: LAB 色彩空間分析（移除亮度不匹配） ===
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

        result = cv2.cvtColor(lab_result.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(
            np.float32
        )

        # === 方法 3: YCrCb 色度分離（新增，針對色邊特別有效） ===
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
        result = cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB).astype(np.float32)

        # === 方法 4: 基於 alpha 的分層處理 ===
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

        # === 方法 5: 局部自適應校正（基於鄰域） ===
        # 對每個邊緣像素，參考其鄰近的確定前景色
        if np.any(edge_mask):
            # 膨脹邊緣遮罩找到鄰域
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel).astype(bool)

            # 確定前景區域
            certain_foreground = (
                alpha_normalized > ALPHA_THRESHOLD_CERTAIN_FOREGROUND
            ) & ~edge_dilated

            if np.any(certain_foreground):
                # 使用確定前景的平均色進行引導
                fg_mean_color = np.mean(
                    result[certain_foreground].reshape(-1, 3), axis=0
                )

                # 在邊緣區域向前景色微調（保留細節）
                blend_factor = defringe_strength * 0.15
                result[edge_mask] = result[edge_mask] * (1 - blend_factor) + (
                    fg_mean_color * blend_factor
                )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_color_filter(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        階段 4: 色彩過濾（針對純色背景）

        綠色模式：只修改 RGB（不碰 alpha），避免過度移除。
        黑/白模式：修改 alpha mask。

        Args:
            image: RGB 圖片（會被 in-place 修改）
            alpha: 當前 alpha matte

        Returns:
            精煉後的 alpha（綠色模式下不變）
        """
        if not self.color_filter.enabled or self.color_filter.color == ColorFilter.NONE:
            return alpha

        logger.debug("Applying color filter: %s", self.color_filter.color.value)

        if self.color_filter.color == ColorFilter.GREEN:
            # 綠色模式：純 RGB despill，不修改 alpha
            self._apply_green_despill(image, alpha)
            return alpha

        # 黑/白模式：使用 alpha mask
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        if self.color_filter.color == ColorFilter.BLACK:
            l_channel = lab[:, :, 0]
            color_mask = (l_channel < BLACK_THRESHOLD).astype(
                np.uint8
            ) * PIXEL_MAX_VALUE

        elif self.color_filter.color == ColorFilter.WHITE:
            l_channel = lab[:, :, 0]
            color_mask = (l_channel > WHITE_THRESHOLD).astype(
                np.uint8
            ) * PIXEL_MAX_VALUE

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

    @staticmethod
    def _smooth_alpha_edges(alpha: np.ndarray, image: np.ndarray) -> np.ndarray:
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

    def _apply_green_despill(self, image: np.ndarray, alpha: np.ndarray) -> None:
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
        """
        alpha_norm = alpha.astype(np.float32) / PIXEL_MAX_VALUE
        strength = self.color_filter.edge_refine_strength

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

    @staticmethod
    def _suppress_residual_green(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        階段 5.5: 去污染後的殘留綠色抑制

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

    def process(self, input_path: Path, output_path: Path) -> bool:
        """
        處理單張圖片（完整多階段流程）

        Args:
            input_path: 輸入圖片路徑
            output_path: 輸出圖片路徑

        Returns:
            處理是否成功
        """
        self.ensure_model_loaded()

        try:
            # 載入圖片
            original = Image.open(input_path).convert("RGB")
            image_np = np.array(original)
            logger.debug("Processing: %s (%dx%d)", input_path.name, *original.size)

            # 階段 1: RMBG-2.0 主分割
            alpha = self._apply_rmbg_segmentation(original)
            logger.debug("Stage 1: RMBG-2.0 segmentation complete")

            # 階段 2: Trimap refinement（如果啟用）
            if self.use_trimap_refine:
                # 動態計算 trimap 參數（考慮解析度和強度）
                # 1. 根據解析度調整基礎核大小
                base_erode = self._calculate_trimap_kernel_size(
                    original.size, self.TRIMAP_ERODE_KERNEL
                )
                base_dilate = self._calculate_trimap_kernel_size(
                    original.size, self.TRIMAP_DILATE_KERNEL
                )

                # 2. 根據強度微調（強度越高，核越小，未知區越窄）
                erode_k = int(base_erode * (1.5 - self.strength * 0.5))
                dilate_k = int(base_dilate * (1.5 - self.strength * 0.5))

                # 確保至少為 3（奇數）
                erode_k = max(3, erode_k if erode_k % 2 == 1 else erode_k + 1)
                dilate_k = max(3, dilate_k if dilate_k % 2 == 1 else dilate_k + 1)

                trimap = self._create_trimap(alpha, erode_k, dilate_k)
                alpha = self._refine_alpha_in_unknown_region(image_np, alpha, trimap)
                logger.debug(
                    "Stage 2: Trimap refinement complete (kernel: %d/%d)",
                    erode_k,
                    dilate_k,
                )

            # 階段 3: Advanced defringing
            # 綠色模式已有 despill，跳過通用去色邊（避免與 despill 疊加造成紫色）
            if not (
                self.color_filter.enabled
                and self.color_filter.color == ColorFilter.GREEN
            ):
                image_np = self._apply_advanced_defringing(image_np, alpha)
                logger.debug("Stage 3: Advanced defringing complete")
            else:
                logger.debug("Stage 3: Skipped (green despill handles defringing)")

            # 階段 4: 色彩過濾（如果啟用）
            alpha = self._apply_color_filter(image_np, alpha)
            logger.debug("Stage 4: Color filter complete")

            # 階段 5: 邊緣去污染（修復背景色滲透）
            if self.alpha_config.edge_decontamination:
                decon_strength = self.alpha_config.decontamination_strength
                # 綠色模式已有 despill，降低去污染強度避免紫色過校正
                if (
                    self.color_filter.enabled
                    and self.color_filter.color == ColorFilter.GREEN
                ):
                    decon_strength = min(decon_strength, 0.15)
                image_np = decontaminate_edges(
                    image_np,
                    alpha,
                    decon_strength,
                    use_kmeans=True,  # 使用 KMeans 智能背景色估計
                )
                logger.debug(
                    "Stage 5: Edge decontamination complete (strength=%.2f)",
                    decon_strength,
                )

            # 階段 5.5: 殘留綠色抑制（去污染後的最終清理）
            if (
                self.color_filter.enabled
                and self.color_filter.color == ColorFilter.GREEN
            ):
                image_np = self._suppress_residual_green(image_np, alpha)
                logger.debug("Stage 5.5: Residual green suppression complete")

            # 階段 6: Alpha 模式處理
            if self.alpha_config.mode.value == "premultiplied":
                image_np = premultiply_alpha(image_np, alpha)
                logger.debug("Stage 6: Converted to premultiplied alpha")

            # 組合 RGBA
            rgba = np.dstack([image_np, alpha])
            result = Image.fromarray(rgba, "RGBA")

            # 儲存
            result.save(output_path, "PNG", optimize=True)
            logger.debug("Saved: %s", output_path.name)

        except Exception:
            logger.exception("Ultra backend failed: %s", input_path.name)
            return False
        else:
            return True

    @classmethod
    def from_preset(
        cls,
        level: PresetLevel | str,
        color_filter: ColorFilterConfig | None = None,
        device: str | None = None,
    ) -> "UltraBackend":
        """
        從預設配置建立 Ultra Backend

        Args:
            level: 預設等級（balanced / high / ultra）
            color_filter: 可選的色彩過濾設定（覆蓋預設）
            device: 計算設備（覆蓋預設）

        Returns:
            配置好的 UltraBackend 實例

        Examples:
            >>> # 使用中等預設
            >>> backend = UltraBackend.from_preset("balanced")
            >>>
            >>> # 使用高品質預設 + 綠幕過濾
            >>> from src.common import ColorFilter, ColorFilterConfig
            >>> green_filter = ColorFilterConfig(color=ColorFilter.GREEN, enabled=True)
            >>> backend = UltraBackend.from_preset("high", color_filter=green_filter)
            >>>
            >>> # 使用最強預設
            >>> backend = UltraBackend.from_preset(PresetLevel.ULTRA)
        """
        # 取得預設配置
        preset = get_preset(level)

        # 覆蓋色彩過濾（如果提供）
        final_color_filter = color_filter or preset.color_filter

        logger.info("Using preset: %s", preset.level_name)

        # 建立實例
        return cls(
            strength=preset.strength,
            color_filter=final_color_filter,
            use_trimap_refine=preset.use_trimap_refine,
            alpha_config=preset.to_alpha_config(),
            resolution_config=preset.to_resolution_config(),
            device=device,
        )

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模型列表"""
        return ["auto"]

    @classmethod
    def get_available_presets(cls) -> dict[str, str]:
        """
        取得可用的預設配置列表

        Returns:
            預設名稱 -> 描述的字典
        """
        from src.common.preset_config import list_presets

        return list_presets()

    @classmethod
    def get_model_description(cls) -> str:
        """取得模型說明"""
        return """
  極致背景移除方案（非商用）：

    auto - BRIA RMBG-2.0 自動配置
           ⚠️  僅供非商用使用（CC BY-NC 4.0）
           • 專業級 alpha matte 輸出
           • Trimap-based edge refinement
           • Portrait matting refinement (人像/頭髮優化)
           • Multi-space defringing
           • KMeans 智能背景色估計
           • 可選純色背景過濾

  推薦設定：
    • 一般圖片：強度 0.6-0.7
    • 複雜邊緣（頭髮）：強度 0.7-0.9 + 人像精修
    • 純色背景：強度 0.8 + 色彩過濾

  ⚠️  商業使用需與 BRIA 協議
"""
