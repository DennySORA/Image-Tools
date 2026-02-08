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
from transformers import AutoModelForImageSegmentation

from src.backends.registry import BackendRegistry
from src.common import ColorFilter, ColorFilterConfig
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
    INPUT_SIZE: ClassVar[tuple[int, int]] = (1024, 1024)

    # Trimap 參數
    TRIMAP_ERODE_KERNEL: ClassVar[int] = 10  # 腐蝕核大小
    TRIMAP_DILATE_KERNEL: ClassVar[int] = 10  # 膨脹核大小

    def __init__(
        self,
        model: str = "auto",  # noqa: ARG002
        strength: float = 0.8,
        color_filter: ColorFilterConfig | None = None,
        use_trimap_refine: bool = True,
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
            device: 計算設備（cuda/cpu），None 則自動選擇
        """
        super().__init__(strength=strength)
        self.color_filter = color_filter or ColorFilterConfig()
        self.use_trimap_refine = use_trimap_refine

        # 設備配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 模型和轉換器
        self._model: Any = None
        self._transform: transforms.Compose | None = None

        logger.info("Ultra backend initialized (NON-COMMERCIAL USE ONLY)")
        logger.info("  Model: BRIA RMBG-2.0")
        logger.info("  Device: %s", self.device)
        logger.info("  Strength: %.2f", self.strength)
        logger.info("  Trimap refinement: %s", self.use_trimap_refine)
        logger.info("  Color filter: %s", self.color_filter.color.value)

    def load_model(self) -> None:
        """載入 RMBG-2.0 模型"""
        logger.info("Loading BRIA RMBG-2.0 model...")
        logger.info("⚠️  This model is for NON-COMMERCIAL use only (CC BY-NC 4.0)")

        # 載入模型
        self._model = AutoModelForImageSegmentation.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )
        self._model.to(self.device)
        self._model.eval()

        # 設置圖像轉換
        self._transform = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        logger.info("RMBG-2.0 model loaded successfully")

    def _apply_rmbg_segmentation(self, image: Image.Image) -> np.ndarray:
        """
        階段 1: 使用 RMBG-2.0 進行主要分割

        Args:
            image: 輸入圖片 (RGB)

        Returns:
            Alpha matte (0-255 uint8 numpy array)
        """
        if self._model is None or self._transform is None:
            raise RuntimeError("Model not loaded")

        # 保存原始尺寸
        original_size = image.size

        # 轉換並推論
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model(input_tensor)[0][0]

        # 轉為 numpy 並調整大小回原圖
        alpha = output.cpu().numpy()
        alpha = (alpha * PIXEL_MAX_VALUE).astype(np.uint8)

        # Resize 回原始尺寸
        return cv2.resize(alpha, original_size, interpolation=cv2.INTER_LINEAR)  # type: ignore[no-any-return]

    def _create_trimap(
        self, alpha: np.ndarray, erode_kernel: int, dilate_kernel: int
    ) -> np.ndarray:
        """
        建立 trimap（三值圖）

        Args:
            alpha: Alpha matte (0-255)
            erode_kernel: 腐蝕核大小
            dilate_kernel: 膨脹核大小

        Returns:
            Trimap: 0=背景, 128=未知區, 255=前景
        """
        # 二值化
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        # 腐蝕得到確定前景
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel)
        )
        foreground = cv2.erode(binary, kernel_erode, iterations=1)

        # 膨脹得到確定背景（反向）
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)
        )
        background_inv = cv2.dilate(binary, kernel_dilate, iterations=1)
        background = cv2.bitwise_not(background_inv)

        # 組合 trimap
        trimap = np.full_like(alpha, TRIMAP_UNKNOWN)  # 預設未知
        trimap[background == PIXEL_MAX_VALUE] = 0  # 確定背景
        trimap[foreground == PIXEL_MAX_VALUE] = PIXEL_MAX_VALUE  # 確定前景

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
        refined_alpha = alpha.copy().astype(np.float32) / PIXEL_MAX_VALUE

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

        # 再做一次邊緣柔化（只在未知區附近）
        edge_mask = cv2.dilate(
            unknown_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)

        if np.any(edge_mask):
            blur_size = int(3 + self.strength * 4)  # 3-7
            if blur_size % 2 == 0:
                blur_size += 1

            blurred = cv2.GaussianBlur(refined_alpha, (blur_size, blur_size), 0)
            refined_alpha[edge_mask] = blurred[edge_mask]

        return (refined_alpha * PIXEL_MAX_VALUE).astype(np.uint8)

    def _apply_advanced_defringing(
        self, image: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """
        階段 3: 進階去色邊（多色彩空間分析）

        使用 RGB + LAB + HSV 多重分析來徹底移除色彩污染

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

        # 找出邊緣區域（半透明像素）
        edge_mask = (alpha_normalized > EDGE_ALPHA_MIN) & (alpha_normalized < EDGE_ALPHA_MAX)

        if not np.any(edge_mask):
            return image

        # === 方法 1: RGB 色彩平衡 ===
        r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        color_mean = (r + g + b) / 3

        # 計算各通道偏離平均值的程度
        r_diff = r - color_mean
        g_diff = g - color_mean
        b_diff = b - color_mean

        # 在邊緣區域減少色偏
        defringe_strength = (self.strength - 0.4) * 1.5  # 0->0.9
        result[:, :, 0][edge_mask] -= r_diff[edge_mask] * defringe_strength * 0.6
        result[:, :, 1][edge_mask] -= g_diff[edge_mask] * defringe_strength * 0.6
        result[:, :, 2][edge_mask] -= b_diff[edge_mask] * defringe_strength * 0.6

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

        # === 方法 3: 基於 alpha 的色彩混合調整 ===
        # 對於非常半透明的邊緣（alpha < 0.5），更激進地去色
        very_transparent_mask = edge_mask & (alpha_normalized < 0.5)  # noqa: PLR2004
        if np.any(very_transparent_mask):
            # 向灰階混合
            gray_value = np.mean(result, axis=2, keepdims=True)
            result[very_transparent_mask] = (
                result[very_transparent_mask] * 0.7
                + gray_value[very_transparent_mask] * 0.3
            )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_color_filter(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        階段 4: 色彩過濾（針對純色背景）

        Args:
            image: RGB 圖片
            alpha: 當前 alpha matte

        Returns:
            精煉後的 alpha
        """
        if not self.color_filter.enabled or self.color_filter.color == ColorFilter.NONE:
            return alpha

        logger.debug("Applying color filter: %s", self.color_filter.color.value)

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        # 建立色彩遮罩
        if self.color_filter.color == ColorFilter.GREEN:
            # 綠幕檢測
            lower = np.array([self.color_filter.hue_range[0], 40, 40])
            upper = np.array([self.color_filter.hue_range[1], 255, 255])
            color_mask = cv2.inRange(hsv, lower, upper)

            # Despill 綠色
            r, g, b = (
                image[:, :, 0].astype(np.float32),
                image[:, :, 1].astype(np.float32),
                image[:, :, 2].astype(np.float32),
            )
            rb_avg = (r + b) / 2
            green_excess = np.maximum(g - rb_avg, 0)
            image[:, :, 1] = np.clip(
                g - green_excess * self.color_filter.edge_refine_strength, 0, 255
            ).astype(np.uint8)

        elif self.color_filter.color == ColorFilter.BLACK:
            l_channel = lab[:, :, 0]
            color_mask = (l_channel < BLACK_THRESHOLD).astype(np.uint8) * PIXEL_MAX_VALUE

        elif self.color_filter.color == ColorFilter.WHITE:
            l_channel = lab[:, :, 0]
            color_mask = (l_channel > WHITE_THRESHOLD).astype(np.uint8) * PIXEL_MAX_VALUE

        else:
            return alpha

        # 優化遮罩
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # 腐蝕以避免顏色邊
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.erode(color_mask, erode_kernel, iterations=1)

        # 羽化
        color_mask = cv2.GaussianBlur(color_mask, (7, 7), 0)

        # 與現有 alpha 合併（取最小值）
        foreground_mask = PIXEL_MAX_VALUE - color_mask
        return np.minimum(alpha, foreground_mask)  # type: ignore[no-any-return]

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
                # 根據強度動態調整 trimap 參數
                erode_k = int(self.TRIMAP_ERODE_KERNEL * (1.5 - self.strength * 0.5))
                dilate_k = int(self.TRIMAP_DILATE_KERNEL * (1.5 - self.strength * 0.5))

                trimap = self._create_trimap(alpha, erode_k, dilate_k)
                alpha = self._refine_alpha_in_unknown_region(image_np, alpha, trimap)
                logger.debug("Stage 2: Trimap refinement complete")

            # 階段 3: Advanced defringing
            image_np = self._apply_advanced_defringing(image_np, alpha)
            logger.debug("Stage 3: Advanced defringing complete")

            # 階段 4: 色彩過濾（如果啟用）
            alpha = self._apply_color_filter(image_np, alpha)
            logger.debug("Stage 4: Color filter complete")

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
    def get_available_models(cls) -> list[str]:
        """取得可用模型列表"""
        return ["auto"]

    @classmethod
    def get_model_description(cls) -> str:
        """取得模型說明"""
        return """
  極致背景移除方案（非商用）：

    auto - BRIA RMBG-2.0 自動配置
           ⚠️  僅供非商用使用（CC BY-NC 4.0）
           • 專業級 alpha matte 輸出
           • Trimap-based edge refinement
           • Multi-space defringing
           • 可選純色背景過濾

  推薦設定：
    • 一般圖片：強度 0.6-0.7
    • 複雜邊緣（頭髮）：強度 0.7-0.9
    • 純色背景：強度 0.8 + 色彩過濾

  ⚠️  商業使用需與 BRIA 協議
"""
