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
from src.features.background_removal.color_filter import (
    apply_color_filter,
    suppress_residual_green,
)
from src.features.background_removal.defringing import apply_advanced_defringing
from src.features.background_removal.trimap import (
    calculate_trimap_kernel_size,
    create_trimap,
    refine_alpha_in_unknown_region,
)


logger = logging.getLogger(__name__)

# 常數定義
PIXEL_MAX_VALUE = 255


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
                alpha = self._apply_trimap_refinement(original.size, image_np, alpha)

            # 階段 3: Advanced defringing
            image_np = self._apply_defringing_stage(image_np, alpha)

            # 階段 4: 色彩過濾（如果啟用）
            alpha = apply_color_filter(image_np, alpha, self.color_filter)
            logger.debug("Stage 4: Color filter complete")

            # 階段 5: 邊緣去污染（修復背景色滲透）
            image_np = self._apply_decontamination_stage(image_np, alpha)

            # 階段 5.5: 殘留綠色抑制（去污染後的最終清理）
            if (
                self.color_filter.enabled
                and self.color_filter.color == ColorFilter.GREEN
            ):
                image_np = suppress_residual_green(image_np, alpha)
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

    def _apply_trimap_refinement(
        self,
        original_size: tuple[int, int],
        image_np: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """套用 Trimap 精修"""
        # 動態計算 trimap 參數（考慮解析度和強度）
        # 1. 根據解析度調整基礎核大小
        base_erode = calculate_trimap_kernel_size(
            original_size, self.TRIMAP_ERODE_KERNEL
        )
        base_dilate = calculate_trimap_kernel_size(
            original_size, self.TRIMAP_DILATE_KERNEL
        )

        # 2. 根據強度微調（強度越高，核越小，未知區越窄）
        erode_k = int(base_erode * (1.5 - self.strength * 0.5))
        dilate_k = int(base_dilate * (1.5 - self.strength * 0.5))

        # 確保至少為 3（奇數）
        erode_k = max(3, erode_k if erode_k % 2 == 1 else erode_k + 1)
        dilate_k = max(3, dilate_k if dilate_k % 2 == 1 else dilate_k + 1)

        trimap = create_trimap(alpha, erode_k, dilate_k)
        alpha = refine_alpha_in_unknown_region(image_np, alpha, trimap, self.strength)
        logger.debug(
            "Stage 2: Trimap refinement complete (kernel: %d/%d)",
            erode_k,
            dilate_k,
        )
        return alpha

    def _apply_defringing_stage(
        self, image_np: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """套用去色邊階段"""
        # 綠色模式已有 despill，跳過通用去色邊（避免與 despill 疊加造成紫色）
        if not (
            self.color_filter.enabled and self.color_filter.color == ColorFilter.GREEN
        ):
            image_np = apply_advanced_defringing(image_np, alpha, self.strength)
            logger.debug("Stage 3: Advanced defringing complete")
        else:
            logger.debug("Stage 3: Skipped (green despill handles defringing)")
        return image_np

    def _apply_decontamination_stage(
        self, image_np: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """套用邊緣去污染階段"""
        if not self.alpha_config.edge_decontamination:
            return image_np

        decon_strength = self.alpha_config.decontamination_strength
        # 綠色模式已有 despill，降低去污染強度避免紫色過校正
        if self.color_filter.enabled and self.color_filter.color == ColorFilter.GREEN:
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
        return image_np

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
