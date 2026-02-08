"""
人像 Matting 精修模組

提供專門針對人像/頭髮的第二階段 Alpha Matting 精修
支持 MODNet 等專業人像 matting 模型
"""

import logging
from typing import Any

import cv2
import numpy as np
import torch


logger = logging.getLogger(__name__)

# 常數定義
EDGE_DETECTION_THRESHOLD = 0.1  # 邊緣檢測閾值
HAIR_TEXTURE_THRESHOLD = 5  # 頭髮紋理檢測閾值


class PortraitMattingRefiner:
    """
    人像 Matting 精修器

    使用 MODNet 或其他人像專用 matting 模型對邊緣進行精修
    特別適合處理頭髮、毛髮等複雜半透明邊界
    """

    def __init__(
        self,
        model_name: str = "modnet",
        device: str | None = None,
        enable_hr_mode: bool = False,
    ):
        """
        初始化人像 matting 精修器

        Args:
            model_name: 模型名稱 ("modnet" 或 "enhanced")
            device: 計算設備（cuda/cpu），None 則自動選擇
            enable_hr_mode: 是否啟用高解析度模式（更慢但更精確）
        """
        self.model_name = model_name
        self.enable_hr_mode = enable_hr_mode

        # 設備配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model: Any = None
        self._model_loaded = False

        logger.info("Portrait matting refiner initialized")
        logger.info("  Model: %s", self.model_name)
        logger.info("  Device: %s", self.device)
        logger.info("  High-res mode: %s", self.enable_hr_mode)

    def load_model(self) -> None:
        """載入人像 matting 模型"""
        if self._model_loaded:
            return

        if self.model_name == "modnet":
            self._load_modnet()
        elif self.model_name == "enhanced":
            # Enhanced mode 使用進階影像處理技術，不需要額外模型
            logger.info("Enhanced portrait matting mode (no model needed)")
            self._model_loaded = True
        else:
            msg = f"Unknown portrait matting model: {self.model_name}"
            raise ValueError(msg)

    def _load_modnet(self) -> None:
        """
        載入 MODNet 模型

        注意: 需要安裝 MODNet 依賴和下載模型權重
        如果模型不可用，將回退到 enhanced 模式
        """
        try:
            # 嘗試導入 MODNet（如果已安裝）
            # 這裡使用動態導入，以避免強制依賴
            # from modnet import MODNet  # type: ignore

            # TODO: 實現 MODNet 載入邏輯
            # 需要下載預訓練模型: modnet_photographic_portrait_matting.ckpt
            # 模型大小約 27MB

            logger.warning(
                "MODNet integration is not yet implemented. "
                "Falling back to enhanced mode."
            )
            self.model_name = "enhanced"
            self._model_loaded = True

        except ImportError:
            logger.warning(
                "MODNet not installed. Install with: pip install modnet-torch"
            )
            logger.info("Falling back to enhanced portrait matting mode")
            self.model_name = "enhanced"
            self._model_loaded = True

    def refine_alpha(
        self,
        image: np.ndarray,
        initial_alpha: np.ndarray,
        focus_strength: float = 0.7,
    ) -> np.ndarray:
        """
        精修 alpha matte（專注於人像/頭髮）

        Args:
            image: RGB 圖片 (H, W, 3), uint8
            initial_alpha: 初始 alpha matte (H, W), uint8
            focus_strength: 精修強度 (0.0-1.0)

        Returns:
            精修後的 alpha matte (H, W), uint8
        """
        self.load_model()

        if self.model_name == "modnet":
            return self._refine_with_modnet(image, initial_alpha, focus_strength)
        return self._refine_enhanced(image, initial_alpha, focus_strength)

    def _refine_with_modnet(
        self,
        image: np.ndarray,  # noqa: ARG002
        initial_alpha: np.ndarray,
        focus_strength: float,  # noqa: ARG002
    ) -> np.ndarray:
        """
        使用 MODNet 精修 alpha

        Args:
            image: RGB 圖片
            initial_alpha: 初始 alpha
            focus_strength: 精修強度

        Returns:
            精修後的 alpha
        """
        # TODO: 實現 MODNet 推論邏輯
        logger.debug("MODNet refinement (placeholder)")
        return initial_alpha

    def _refine_enhanced(
        self,
        image: np.ndarray,
        initial_alpha: np.ndarray,
        focus_strength: float,
    ) -> np.ndarray:
        """
        使用增強影像處理技術精修 alpha（無需額外模型）

        這個方法使用多種影像處理技術來優化人像邊緣：
        1. 邊緣檢測與增強
        2. 基於梯度的細節保留
        3. 自適應平滑（保留頭髮等細節）
        4. 色彩引導的邊界精修

        Args:
            image: RGB 圖片
            initial_alpha: 初始 alpha
            focus_strength: 精修強度

        Returns:
            精修後的 alpha
        """
        logger.debug("Enhanced portrait matting refinement")

        alpha_float = initial_alpha.astype(np.float32) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 1. 檢測邊緣（找出需要精修的區域）
        edges = self._detect_portrait_edges(gray, alpha_float)

        # 2. 在邊緣區域應用自適應處理
        refined_alpha = self._adaptive_edge_refinement(
            image, alpha_float, edges, focus_strength
        )

        # 3. 頭髮/毛髮細節增強
        refined_alpha = self._enhance_hair_details(
            gray, refined_alpha, edges, focus_strength
        )

        # 4. 最終平滑（保持細節）
        refined_alpha = self._detail_preserving_smooth(refined_alpha, edges)

        return (refined_alpha * 255).astype(np.uint8)

    def _detect_portrait_edges(self, gray: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        檢測人像邊緣區域

        Args:
            gray: 灰階圖片 (H, W), uint8
            alpha: Alpha matte (H, W), float32 [0, 1]

        Returns:
            邊緣遮罩 (H, W), bool
        """
        # Alpha 梯度（找出過渡區域）
        alpha_grad = np.gradient(alpha)
        alpha_edge = np.sqrt(alpha_grad[0] ** 2 + alpha_grad[1] ** 2)

        # 影像邊緣（Canny）
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_canny = edges_canny.astype(np.float32) / 255.0

        # 結合兩種邊緣
        combined_edges = np.maximum(alpha_edge, edges_canny)

        # 擴展邊緣區域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        edge_mask = cv2.dilate(
            (combined_edges > EDGE_DETECTION_THRESHOLD).astype(np.uint8), kernel
        )

        return edge_mask.astype(bool)

    def _adaptive_edge_refinement(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
        edges: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """
        自適應邊緣精修

        Args:
            image: RGB 圖片
            alpha: Alpha matte (float32)
            edges: 邊緣遮罩
            strength: 精修強度

        Returns:
            精修後的 alpha
        """
        refined = alpha.copy()

        if not np.any(edges):
            return refined

        # 在邊緣區域使用雙邊濾波（保留邊緣細節）
        if np.any(edges):
            # 只對邊緣區域做處理
            alpha_edge_region = refined[edges]

            # 使用導向濾波（以原圖為引導）
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

            # 動態半徑
            radius = int(5 + strength * 10)

            filtered = cv2.ximgproc.guidedFilter(
                guide=gray, src=refined, radius=radius, eps=0.001
            )

            # 根據強度混合原始和濾波結果
            blend_factor = strength * 0.6
            refined[edges] = (
                alpha_edge_region * (1 - blend_factor) + filtered[edges] * blend_factor
            )

        return refined

    def _enhance_hair_details(
        self,
        gray: np.ndarray,
        alpha: np.ndarray,
        edges: np.ndarray,  # noqa: ARG002
        strength: float,
    ) -> np.ndarray:
        """
        增強頭髮/毛髮細節

        Args:
            gray: 灰階圖片
            alpha: Alpha matte (float32)
            edges: 邊緣遮罩
            strength: 精修強度

        Returns:
            增強後的 alpha
        """
        enhanced = alpha.copy()

        # 檢測細微紋理（可能是頭髮）
        # 使用高斯差分（DoG）檢測細節
        blur1 = cv2.GaussianBlur(gray, (3, 3), 1.0)
        blur2 = cv2.GaussianBlur(gray, (9, 9), 3.0)
        dog = np.abs(blur1.astype(np.float32) - blur2.astype(np.float32))

        # 在半透明區域，如果有細節紋理，增強 alpha
        semi_transparent = (alpha > 0.1) & (alpha < 0.9)  # noqa: PLR2004
        hair_candidate = semi_transparent & (
            dog > HAIR_TEXTURE_THRESHOLD
        )  # 有細節紋理的半透明區域

        if np.any(hair_candidate):
            # 輕微增強這些區域的 alpha（讓細節更明顯）
            enhancement = dog[hair_candidate] / 255.0 * strength * 0.3
            enhanced[hair_candidate] = np.clip(
                alpha[hair_candidate] + enhancement, 0, 1
            )

        return enhanced

    def _detail_preserving_smooth(
        self, alpha: np.ndarray, edges: np.ndarray
    ) -> np.ndarray:
        """
        保留細節的平滑處理

        Args:
            alpha: Alpha matte (float32)
            edges: 邊緣遮罩

        Returns:
            平滑後的 alpha
        """
        # 對非邊緣區域做輕微平滑
        non_edges = ~edges

        if np.any(non_edges):
            smoothed = cv2.GaussianBlur(alpha, (5, 5), 1.0)
            alpha[non_edges] = smoothed[non_edges]

        return alpha


def refine_portrait_alpha(
    image: np.ndarray,
    alpha: np.ndarray,
    model_name: str = "enhanced",
    strength: float = 0.7,
    device: str | None = None,
) -> np.ndarray:
    """
    便捷函數：對人像 alpha 進行精修

    Args:
        image: RGB 圖片 (H, W, 3), uint8
        alpha: Alpha matte (H, W), uint8
        model_name: 模型名稱 ("modnet" 或 "enhanced")
        strength: 精修強度 (0.0-1.0)
        device: 計算設備

    Returns:
        精修後的 alpha matte (H, W), uint8
    """
    refiner = PortraitMattingRefiner(model_name=model_name, device=device)
    return refiner.refine_alpha(image, alpha, strength)
