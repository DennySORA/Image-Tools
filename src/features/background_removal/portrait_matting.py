"""
人像 Matting 精修模組

提供專門針對人像/頭髮的第二階段 Alpha Matting 精修
支持 BiRefNet-matting 和增強影像處理模式

BiRefNet-matting 參考：
- Hugging Face: https://huggingface.co/ZhengPeng7/BiRefNet-matting
- GitHub: https://github.com/ZhengPeng7/BiRefNet
- 授權：MIT License
"""

import logging
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms  # type: ignore[import-untyped]


logger = logging.getLogger(__name__)

# BiRefNet 模型配置
BIREFNET_MODELS: dict[str, str] = {
    "birefnet": "ZhengPeng7/BiRefNet-matting",
    "birefnet-hr": "ZhengPeng7/BiRefNet_HR-matting",
    "birefnet-portrait": "ZhengPeng7/BiRefNet-portrait",
}
BIREFNET_INPUT_SIZES: dict[str, tuple[int, int]] = {
    "birefnet": (1024, 1024),
    "birefnet-hr": (2048, 2048),
    "birefnet-portrait": (1024, 1024),
}

# 常數定義
EDGE_DETECTION_THRESHOLD = 0.1  # 邊緣檢測閾值
HAIR_TEXTURE_THRESHOLD = 5  # 頭髮紋理檢測閾值
SKIN_HUE_MIN = 0  # 肤色 Hue 最小值
SKIN_HUE_MAX = 50  # 肤色 Hue 最大值（偏橙色）
SKIN_SAT_MIN = 0.1  # 肤色饱和度最小值
SKIN_SAT_MAX = 0.7  # 肤色饱和度最大值
SKIN_VAL_MIN = 0.2  # 肤色明度最小值


class PortraitMattingRefiner:
    """
    人像 Matting 精修器

    使用 BiRefNet-matting 或增強影像處理技術對邊緣進行精修
    特別適合處理頭髮、毛髮等複雜半透明邊界

    支援模型：
    - birefnet: BiRefNet-matting（通用 matting，1024x1024，MIT 授權）
    - birefnet-hr: BiRefNet_HR-matting（高解析度 2048x2048）
    - birefnet-portrait: BiRefNet-portrait（人像專用）
    - enhanced: 增強影像處理（無需額外模型下載）
    """

    def __init__(
        self,
        model_name: str = "birefnet",
        device: str | None = None,
        enable_hr_mode: bool = False,
    ):
        """
        初始化人像 matting 精修器

        Args:
            model_name: 模型名稱 ("birefnet" / "birefnet-hr" /
                       "birefnet-portrait" / "enhanced")
            device: 計算設備（cuda/mps/cpu），None 則自動選擇
            enable_hr_mode: 是否啟用高解析度模式（使用 birefnet-hr）
        """
        # 向後相容：modnet → birefnet
        if model_name == "modnet":
            model_name = "birefnet"

        self.model_name = model_name
        self.enable_hr_mode = enable_hr_mode

        # 高解析度模式自動切換到 birefnet-hr
        if enable_hr_mode and model_name == "birefnet":
            self.model_name = "birefnet-hr"

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

        self._model: Any = None
        self._model_loaded = False
        self._transform: transforms.Compose | None = None

        logger.info(
            "Portrait matting: model=%s, device=%s", self.model_name, self.device
        )

    def load_model(self) -> None:
        """載入人像 matting 模型"""
        if self._model_loaded:
            return

        if self.model_name in BIREFNET_MODELS:
            self._load_birefnet()
        elif self.model_name == "enhanced":
            # Enhanced mode 使用進階影像處理技術，不需要額外模型
            logger.info("Enhanced portrait matting mode (no model needed)")
            self._model_loaded = True
        else:
            msg = f"Unknown portrait matting model: {self.model_name}"
            raise ValueError(msg)

    def _load_birefnet(self) -> None:
        """
        載入 BiRefNet-matting 模型

        使用 Hugging Face transformers 載入預訓練模型
        如果模型不可用，將回退到 enhanced 模式

        BiRefNet 參考：
        - GitHub: https://github.com/ZhengPeng7/BiRefNet
        - Hugging Face: ZhengPeng7/BiRefNet-matting
        - 授權：MIT License
        """
        from src.common.model_compat import load_pretrained_no_meta

        repo_id = BIREFNET_MODELS[self.model_name]
        input_size = BIREFNET_INPUT_SIZES[self.model_name]

        try:
            logger.info("Loading BiRefNet model: %s ...", repo_id)

            self._model = load_pretrained_no_meta(repo_id)
            self._model.to(self.device)
            self._model.eval()

            # 啟用 TF32（Ampere+ GPU 自動加速 float32 矩陣運算）
            if self.device.type == "cuda":
                torch.set_float32_matmul_precision("high")

            # 建立預處理轉換器
            self._transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self._model_loaded = True
            logger.info("BiRefNet loaded on %s", self.device)

        except Exception as e:
            logger.warning(
                "BiRefNet loading failed: %s. Falling back to enhanced mode.",
                e,
            )
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

        if self.model_name in BIREFNET_MODELS:
            return self._refine_with_birefnet(image, initial_alpha, focus_strength)
        return self._refine_enhanced(image, initial_alpha, focus_strength)

    def _refine_with_birefnet(
        self,
        image: np.ndarray,
        initial_alpha: np.ndarray,
        focus_strength: float,
    ) -> np.ndarray:
        """
        使用 BiRefNet-matting 精修 alpha

        BiRefNet 輸出高品質 alpha matte，特別擅長頭髮和半透明物件

        Args:
            image: RGB 圖片 (H, W, 3), uint8
            initial_alpha: 初始 alpha (H, W), uint8
            focus_strength: 精修強度 (0.0-1.0)

        Returns:
            精修後的 alpha (H, W), uint8
        """
        if self._model is None or self._transform is None:
            logger.warning("BiRefNet model not loaded, using initial alpha")
            return initial_alpha

        try:
            h, w = image.shape[:2]

            # 轉為 PIL Image 以使用 transforms
            pil_image = Image.fromarray(image)

            # 預處理並推論
            input_tensor = self._transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self._model(input_tensor)[-1].sigmoid().cpu()

            # 提取 alpha matte 並 resize 回原始大小
            matte = preds[0].squeeze().numpy()
            matte_resized = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

            # 轉換為 uint8
            birefnet_alpha = (matte_resized * 255).clip(0, 255).astype(np.uint8)

            # 根據強度混合初始 alpha 和 BiRefNet 結果
            initial_f = initial_alpha.astype(np.float32) / 255.0
            birefnet_f = birefnet_alpha.astype(np.float32) / 255.0

            blended = initial_f * (1 - focus_strength) + birefnet_f * focus_strength

            logger.debug(
                "BiRefNet refinement complete (strength: %.2f)", focus_strength
            )
            return (blended * 255).clip(0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning("BiRefNet refinement failed: %s. Using initial alpha.", e)
            return initial_alpha

    def _refine_enhanced(
        self,
        image: np.ndarray,
        initial_alpha: np.ndarray,
        focus_strength: float,
    ) -> np.ndarray:
        """
        使用增強影像處理技術精修 alpha（無需額外模型）

        大幅改進的方法使用多種先進影像處理技術：
        1. 肤色檢測輔助人像識別
        2. 多尺度邊緣檢測與增強
        3. 基於梯度和紋理的細節保留
        4. 色彩引導的自適應邊界精修
        5. 多層次頭髮細節增強
        6. 智能平滑（根據區域特性調整）

        Args:
            image: RGB 圖片
            initial_alpha: 初始 alpha
            focus_strength: 精修強度

        Returns:
            精修後的 alpha
        """
        logger.debug("Enhanced portrait matting refinement (advanced)")

        alpha_float = initial_alpha.astype(np.float32) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # === 階段 1: 增強型邊緣檢測 ===
        edges, skin_mask = self._detect_portrait_edges(image, gray, alpha_float)

        # === 階段 2: 肤色引導的 Alpha 優化 ===
        # 在肤色區域內，alpha 應該接近 1（確定前景）
        if np.any(skin_mask):
            # 在肤色區域微調 alpha（增強確定性）
            skin_boost = 0.1 * focus_strength
            alpha_float[skin_mask] = np.clip(alpha_float[skin_mask] + skin_boost, 0, 1)

        # === 階段 3: 多尺度自適應邊緣精修 ===
        refined_alpha = self._adaptive_edge_refinement(
            image, alpha_float, edges, focus_strength
        )

        # === 階段 4: 增強型頭髮/毛髮細節處理 ===
        refined_alpha = self._enhance_hair_details(
            image, gray, refined_alpha, edges, skin_mask, focus_strength
        )

        # === 階段 5: 邊緣一致性優化 ===
        # 確保邊緣過渡平滑但保留細節
        refined_alpha = self._optimize_edge_consistency(
            refined_alpha, edges, focus_strength
        )

        # === 階段 6: 智能平滑（保持細節） ===
        refined_alpha = self._detail_preserving_smooth(refined_alpha, edges)

        return (refined_alpha * 255).astype(np.uint8)

    def _detect_skin_region(self, image: np.ndarray) -> np.ndarray:
        """
        檢測肤色區域（用於人像識別）

        使用 HSV 和 YCrCb 雙重檢測，提高肤色識別準確度

        Args:
            image: RGB 圖片 (H, W, 3), uint8

        Returns:
            肤色遮罩 (H, W), bool
        """
        # 方法 1: HSV 色彩空間
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 肤色範圍（Hue: 0-50, 偏橙紅色）
        skin_mask_hsv = (
            ((h * 360 >= SKIN_HUE_MIN) & (h * 360 <= SKIN_HUE_MAX))
            & (s >= SKIN_SAT_MIN)
            & (s <= SKIN_SAT_MAX)
            & (v >= SKIN_VAL_MIN)
        )

        # 方法 2: YCrCb 色彩空間（補充檢測）
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

        # 肤色範圍（經驗值）
        skin_mask_ycrcb = (
            (y > 80)  # noqa: PLR2004
            & (cr > 133)  # noqa: PLR2004
            & (cr < 173)  # noqa: PLR2004
            & (cb > 77)  # noqa: PLR2004
            & (cb < 127)  # noqa: PLR2004
        )

        # 結合兩種方法
        skin_mask = skin_mask_hsv | skin_mask_ycrcb

        # 形態學處理（去除噪點）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(
            skin_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        skin_mask = cv2.morphologyEx(skin_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        return skin_mask.astype(bool)

    def _detect_portrait_edges(
        self, image: np.ndarray, gray: np.ndarray, alpha: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        增強型人像邊緣檢測

        結合 alpha 梯度、影像邊緣、肤色檢測，更準確識別人像邊界

        Args:
            image: RGB 圖片 (H, W, 3), uint8
            gray: 灰階圖片 (H, W), uint8
            alpha: Alpha matte (H, W), float32 [0, 1]

        Returns:
            edge_mask: 邊緣遮罩 (H, W), bool
            skin_mask: 肤色遮罩 (H, W), bool（用於後續處理）
        """
        # === 階段 1: Alpha 梯度分析 ===
        grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
        alpha_edge = np.sqrt(grad_x**2 + grad_y**2)

        # === 階段 2: 多尺度 Canny 邊緣 ===
        # 使用兩種閾值捕捉不同強度的邊緣
        edges_strong = cv2.Canny(gray, 100, 200)  # 強邊緣
        edges_weak = cv2.Canny(gray, 30, 100)  # 弱邊緣
        edges_canny = np.maximum(edges_strong, edges_weak).astype(np.float32) / 255.0

        # === 階段 3: 肤色檢測 ===
        skin_mask = self._detect_skin_region(image)

        # === 階段 4: 智能邊緣融合 ===
        # 在肤色區域附近的邊緣更可能是人像邊界
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        skin_dilated = cv2.dilate(skin_mask.astype(np.uint8), kernel_dilate)

        # 結合三種邊緣，在肤色區域附近加權
        combined_edges = np.maximum(alpha_edge, edges_canny)

        # 在肤色區域附近的邊緣提升權重
        combined_edges = np.where(
            skin_dilated > 0, combined_edges * 1.5, combined_edges
        )

        # === 階段 5: 自適應閾值 ===
        # 根據邊緣強度分佈自動選擇閾值
        edge_values = combined_edges[combined_edges > 0]
        if len(edge_values) > 0:
            threshold = np.percentile(edge_values, 60)  # 60分位數
            threshold = max(EDGE_DETECTION_THRESHOLD, min(threshold, 0.3))  # noqa: PLR2004
        else:
            threshold = EDGE_DETECTION_THRESHOLD

        # === 階段 6: 擴展邊緣區域 ===
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        edge_mask = cv2.dilate(
            (combined_edges > threshold).astype(np.uint8), kernel
        ).astype(bool)

        return edge_mask, skin_mask

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

    def _enhance_hair_details(  # noqa: PLR0913
        self,
        image: np.ndarray,
        gray: np.ndarray,
        alpha: np.ndarray,
        edges: np.ndarray,
        skin_mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """
        增強型頭髮/毛髮細節處理

        使用多層次紋理分析和色彩信息來識別和增強頭髮區域

        Args:
            image: RGB 圖片
            gray: 灰階圖片
            alpha: Alpha matte (float32)
            edges: 邊緣遮罩
            skin_mask: 肤色遮罩
            strength: 精修強度

        Returns:
            增強後的 alpha
        """
        enhanced = alpha.copy()

        # === 方法 1: 多尺度紋理檢測（DoG） ===
        # 使用高斯差分（DoG）檢測細節
        blur1 = cv2.GaussianBlur(gray, (3, 3), 1.0)
        blur2 = cv2.GaussianBlur(gray, (9, 9), 3.0)
        blur3 = cv2.GaussianBlur(gray, (15, 15), 5.0)

        # 細尺度紋理（細髮絲）
        dog_fine = np.abs(blur1.astype(np.float32) - blur2.astype(np.float32))
        # 粗尺度紋理（髮束）
        dog_coarse = np.abs(blur2.astype(np.float32) - blur3.astype(np.float32))

        # 結合多尺度
        texture_map = np.maximum(dog_fine, dog_coarse * 0.5)

        # === 方法 2: 色彩分析（頭髮通常是深色） ===
        # 計算明度
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        value_channel = hsv[:, :, 2]  # V 通道

        # 暗色區域更可能是頭髮（但不是黑色背景）
        dark_mask = (value_channel < 0.6) & (alpha > 0.1)  # noqa: PLR2004

        # === 方法 3: 空間關係（頭髮通常在肤色區域附近） ===
        if np.any(skin_mask):
            # 膨脹肤色區域以包含頭髮可能範圍
            kernel_hair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            hair_region_mask = cv2.dilate(skin_mask.astype(np.uint8), kernel_hair)

            # 但排除肤色本身
            hair_region_mask = hair_region_mask.astype(bool) & ~skin_mask
        else:
            hair_region_mask = np.ones_like(alpha, dtype=bool)

        # === 階段 4: 識別頭髮候選區域 ===
        semi_transparent = (alpha > 0.05) & (alpha < 0.95)  # noqa: PLR2004

        # 頭髮必須滿足：
        # 1. 半透明
        # 2. 有紋理
        # 3. 暗色或在邊緣
        # 4. 在肤色附近
        hair_candidate = (
            semi_transparent
            & (texture_map > HAIR_TEXTURE_THRESHOLD)
            & (dark_mask | edges)
            & hair_region_mask
        )

        # === 階段 5: 增強頭髮區域的 alpha ===
        if np.any(hair_candidate):
            # 根據紋理強度和位置調整增強程度
            # 紋理越強，增強越多
            texture_strength = np.clip(texture_map / 50.0, 0, 1)

            # 離肤色越近，增強越保守
            if np.any(skin_mask):
                dist_transform = cv2.distanceTransform(
                    (~skin_mask).astype(np.uint8), cv2.DIST_L2, 5
                )
                distance_factor = np.clip(dist_transform / 50.0, 0, 1)
            else:
                distance_factor = np.ones_like(alpha)

            # 計算增強量
            enhancement = (
                texture_strength * distance_factor * strength * 0.25
            )  # 保守增強

            # 應用增強
            enhanced[hair_candidate] = np.clip(
                alpha[hair_candidate] + enhancement[hair_candidate], 0, 1
            )

        # === 階段 6: 頭髮邊緣柔化 ===
        # 對識別出的頭髮區域，做輕微的邊緣柔化
        if np.any(hair_candidate):
            kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            hair_dilated = cv2.dilate(hair_candidate.astype(np.uint8), kernel_smooth)

            smoothed = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
            blend_factor = 0.3 * strength

            enhanced = np.where(
                hair_dilated.astype(bool),
                enhanced * (1 - blend_factor) + smoothed * blend_factor,
                enhanced,
            )

        return enhanced

    def _optimize_edge_consistency(
        self, alpha: np.ndarray, edges: np.ndarray, strength: float
    ) -> np.ndarray:
        """
        優化邊緣一致性

        確保邊緣過渡平滑，同時保留重要細節

        Args:
            alpha: Alpha matte (float32)
            edges: 邊緣遮罩
            strength: 精修強度

        Returns:
            優化後的 alpha
        """
        optimized = alpha.copy()

        if not np.any(edges):
            return optimized

        # === 階段 1: 識別邊緣中的異常值 ===
        # 計算邊緣區域的局部方差
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        local_mean = cv2.filter2D(alpha, -1, kernel)
        local_variance = cv2.filter2D((alpha - local_mean) ** 2, -1, kernel)

        # 高方差區域表示不一致的邊緣
        high_variance_mask = edges & (local_variance > 0.02)  # noqa: PLR2004

        # === 階段 2: 對不一致區域應用雙邊濾波 ===
        if np.any(high_variance_mask):
            # 雙邊濾波保留邊緣但平滑噪聲
            d = int(5 + strength * 5)  # 鄰域直徑
            sigma_color = 0.1 * strength
            sigma_space = d

            filtered = (
                cv2.bilateralFilter(
                    (alpha * 255).astype(np.uint8), d, sigma_color * 100, sigma_space
                ).astype(np.float32)
                / 255.0
            )

            # 只在高方差區域替換
            blend_factor = strength * 0.5
            optimized[high_variance_mask] = (
                alpha[high_variance_mask] * (1 - blend_factor)
                + filtered[high_variance_mask] * blend_factor
            )

        # === 階段 3: 梯度平滑（確保過渡自然） ===
        # 計算梯度
        grad_x = cv2.Sobel(optimized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(optimized, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # 識別梯度過大的區域（不自然的跳變）
        sharp_edges = edges & (gradient_mag > 0.3)  # noqa: PLR2004

        if np.any(sharp_edges):
            # 對過於尖銳的邊緣做柔化
            smoothed = cv2.GaussianBlur(optimized, (5, 5), 1.5)
            optimized[sharp_edges] = (
                optimized[sharp_edges] * 0.6 + smoothed[sharp_edges] * 0.4
            )

        return optimized

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
        model_name: 模型名稱 ("birefnet" / "birefnet-hr" /
                   "birefnet-portrait" / "enhanced")
        strength: 精修強度 (0.0-1.0)
        device: 計算設備

    Returns:
        精修後的 alpha matte (H, W), uint8
    """
    refiner = PortraitMattingRefiner(model_name=model_name, device=device)
    return refiner.refine_alpha(image, alpha, strength)
