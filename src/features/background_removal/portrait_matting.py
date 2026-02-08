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
SKIN_HUE_MIN = 0  # 肤色 Hue 最小值
SKIN_HUE_MAX = 50  # 肤色 Hue 最大值（偏橙色）
SKIN_SAT_MIN = 0.1  # 肤色饱和度最小值
SKIN_SAT_MAX = 0.7  # 肤色饱和度最大值
SKIN_VAL_MIN = 0.2  # 肤色明度最小值


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

        使用 Hugging Face 或 PyTorch Hub 載入預訓練模型
        如果模型不可用，將回退到 enhanced 模式

        MODNet 參考：
        - Paper: https://arxiv.org/abs/2011.11961
        - GitHub: https://github.com/ZHKKKe/MODNet
        - Hugging Face: Xenova/modnet
        """
        try:
            logger.info("Loading MODNet model from Hugging Face...")

            # 方法 1: 嘗試從 Hugging Face 載入（推薦）
            try:
                from huggingface_hub import hf_hub_download

                # 下載 MODNet 權重
                model_path = hf_hub_download(
                    repo_id="Xenova/modnet",
                    filename="onnx/model.onnx",
                    cache_dir=".cache/modnet",
                )

                logger.info("MODNet model downloaded: %s", model_path)

                # 使用 ONNX Runtime 載入
                try:
                    import onnxruntime as ort

                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )

                    providers = ["CPUExecutionProvider"]
                    if self.device.type == "cuda":
                        providers.insert(0, "CUDAExecutionProvider")

                    self._model = ort.InferenceSession(
                        model_path, session_options, providers=providers
                    )
                    self._model_loaded = True
                    logger.info("MODNet loaded successfully (ONNX Runtime)")
                    return

                except ImportError:
                    logger.warning("onnxruntime not installed. Trying PyTorch method.")

            except Exception as e:
                logger.warning("Failed to load from Hugging Face: %s", e)

            # 方法 2: 嘗試從 PyTorch Hub 載入（備選）
            try:
                logger.info("Trying to load MODNet from PyTorch Hub...")

                # 從 GitHub repo 載入
                self._model = torch.hub.load(
                    "ZHKKKe/MODNet",
                    "modnet_photographic_portrait_matting",
                    pretrained=True,
                )
                self._model.to(self.device)
                self._model.eval()
                self._model_loaded = True
                logger.info("MODNet loaded successfully (PyTorch Hub)")
                return

            except Exception as e:
                logger.warning("Failed to load from PyTorch Hub: %s", e)

            # 如果都失敗，回退到 enhanced 模式
            logger.warning(
                "MODNet loading failed. Falling back to enhanced mode. "
                "To use MODNet, install: pip install onnxruntime huggingface-hub"
            )
            self.model_name = "enhanced"
            self._model_loaded = True

        except Exception as e:
            logger.warning(
                "MODNet loading error: %s. Falling back to enhanced mode.", e
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

        if self.model_name == "modnet":
            return self._refine_with_modnet(image, initial_alpha, focus_strength)
        return self._refine_enhanced(image, initial_alpha, focus_strength)

    def _refine_with_modnet(
        self,
        image: np.ndarray,
        initial_alpha: np.ndarray,
        focus_strength: float,
    ) -> np.ndarray:
        """
        使用 MODNet 精修 alpha

        Args:
            image: RGB 圖片 (H, W, 3), uint8
            initial_alpha: 初始 alpha (H, W), uint8
            focus_strength: 精修強度 (0.0-1.0)

        Returns:
            精修後的 alpha (H, W), uint8
        """
        if self._model is None:
            logger.warning("MODNet model not loaded, using initial alpha")
            return initial_alpha

        try:
            h, w = image.shape[:2]

            # MODNet 預處理
            # 調整大小到 512 的倍數（MODNet 要求）
            ref_size = 512
            im_h, im_w = image.shape[:2]
            if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                if im_w >= im_h:
                    im_rh = ref_size
                    im_rw = int(im_w / im_h * ref_size)
                else:
                    im_rw = ref_size
                    im_rh = int(im_h / im_w * ref_size)
            else:
                im_rh = im_h
                im_rw = im_w

            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32

            # Resize 圖片
            image_resized = cv2.resize(
                image, (im_rw, im_rh), interpolation=cv2.INTER_AREA
            )

            # 檢查模型類型
            if hasattr(self._model, "run"):  # ONNX Runtime
                # 轉換為 ONNX 格式 (NCHW)
                input_tensor = (
                    image_resized.astype(np.float32).transpose(2, 0, 1) / 255.0
                )
                input_tensor = np.expand_dims(input_tensor, axis=0)

                # 推論
                input_name = self._model.get_inputs()[0].name
                output_name = self._model.get_outputs()[0].name
                matte = self._model.run([output_name], {input_name: input_tensor})[0]

                # 轉回 numpy (squeeze batch + channel dimensions)
                matte = matte[0, 0]

            else:  # PyTorch 模型
                from torchvision import transforms

                # 轉換為 PyTorch tensor
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )

                input_tensor = transform(image_resized).unsqueeze(0).to(self.device)

                # 推論
                with torch.no_grad():
                    _, _, matte = self._model(input_tensor, True)

                # 轉回 numpy
                matte = matte[0, 0].cpu().numpy()

            # Resize 回原始大小
            matte_resized = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

            # 轉換為 uint8
            modnet_alpha = (matte_resized * 255).clip(0, 255).astype(np.uint8)

            # 根據強度混合初始 alpha 和 MODNet 結果
            initial_alpha_normalized = initial_alpha.astype(np.float32) / 255.0
            modnet_alpha_normalized = modnet_alpha.astype(np.float32) / 255.0

            # 混合：focus_strength 控制 MODNet 的影響力
            blended = (
                initial_alpha_normalized * (1 - focus_strength)
                + modnet_alpha_normalized * focus_strength
            )

            logger.debug("MODNet refinement complete (strength: %.2f)", focus_strength)
            return (blended * 255).clip(0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning("MODNet refinement failed: %s. Using initial alpha.", e)
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
        model_name: 模型名稱 ("modnet" 或 "enhanced")
        strength: 精修強度 (0.0-1.0)
        device: 計算設備

    Returns:
        精修後的 alpha matte (H, W), uint8
    """
    refiner = PortraitMattingRefiner(model_name=model_name, device=device)
    return refiner.refine_alpha(image, alpha, strength)
