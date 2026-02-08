"""
統一背景移除後端 - SOTA 解決方案

基於最新的研究與最佳實踐，整合多階段處理流程：
1. BiRefNet 主模型：高解析度分割，保持邊緣細節
2. Alpha Matting：精細化半透明邊界（頭髮、毛邊）
3. 色彩去污染（Defringing）：移除邊緣顏色滲透
4. 可選色彩過濾：針對純色背景（黑/白/綠）優化

參考文獻：
- BiRefNet: Bilateral Reference for High-Resolution Dichotomous Image Segmentation
- Rembg: 工業級背景移除工具
- Pymatting: Alpha matting 算法實現
"""

import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, cast

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove  # type: ignore[import-untyped]

from src.common import ColorFilter, ColorFilterConfig
from src.core.interfaces import BaseBackend

from .registry import BackendRegistry


logger = logging.getLogger(__name__)

RemoveFunc = Callable[..., bytes]
SessionFactory = Callable[[str], object]


@BackendRegistry.register("unified")
class UnifiedBackend(BaseBackend):
    """
    統一背景移除後端

    整合 SOTA 技術，提供最佳邊緣品質：
    - BiRefNet-massive：最強大的分割模型
    - 自動 Alpha Matting：智能邊緣精細化
    - 可控強度：0.1-1.0 調節處理強度
    - 可選色彩過濾：針對純色背景優化
    """

    name: ClassVar[str] = "unified"
    description: ClassVar[str] = "統一方案 - SOTA BiRefNet + Alpha Matting + 邊緣優化"

    # 使用最強大的 BiRefNet 模型
    PRIMARY_MODEL: ClassVar[str] = "birefnet-massive"

    # Alpha matting 閾值（基於強度動態調整）
    MATTING_STRENGTH_THRESHOLD: ClassVar[float] = 0.3

    def __init__(
        self,
        model: str = "auto",  # 保留 model 參數以符合介面，但內部固定使用 birefnet
        strength: float = 0.7,
        color_filter: ColorFilterConfig | None = None,
    ):
        """
        初始化統一後端

        Args:
            model: 模型名稱（內部固定使用 birefnet-massive，此參數用於相容性）
            strength: 處理強度 (0.1-1.0)
                     - 0.1-0.3: 保守模式，保留更多邊緣像素
                     - 0.4-0.7: 平衡模式（推薦）
                     - 0.8-1.0: 激進模式，最大程度移除背景
            color_filter: 色彩過濾設定（針對純色背景）
        """
        super().__init__(strength=strength)
        self._session: object | None = None
        self.color_filter = color_filter or ColorFilterConfig()

        # 記錄實際使用的模型
        self._actual_model = self.PRIMARY_MODEL
        logger.info("Unified backend initialized with BiRefNet-massive")

    def load_model(self) -> None:
        """載入 BiRefNet 模型"""
        logger.info("Loading unified backend...")
        logger.info("  Primary model: %s", self.PRIMARY_MODEL)
        logger.info("  Strength: %.2f", self.strength)
        logger.info("  Color filter: %s", self.color_filter.color.value)

        # 載入 BiRefNet session
        session_factory = cast(SessionFactory, new_session)
        self._session = session_factory(self.PRIMARY_MODEL)

        logger.info("Unified backend loaded successfully")

    def _calculate_matting_params(
        self,
    ) -> tuple[bool, int, int]:
        """
        根據強度計算 alpha matting 參數

        Returns:
            (是否啟用 matting, 前景閾值, 背景閾值)
        """
        # 強度越高，alpha matting 越激進
        enable_matting = self.strength >= self.MATTING_STRENGTH_THRESHOLD

        if not enable_matting:
            return False, 240, 10

        # 前景閾值：強度越高，閾值越低（更多像素被認為是前景）
        # 範圍：240 (strength=0.3) -> 200 (strength=1.0)
        fg_threshold = int(240 - (self.strength - 0.3) * 60)

        # 背景閾值：強度越高，閾值越高（更嚴格判斷背景）
        # 範圍：10 (strength=0.3) -> 40 (strength=1.0)
        bg_threshold = int(10 + (self.strength - 0.3) * 40)

        return True, fg_threshold, bg_threshold

    def _apply_primary_segmentation(self, image: Image.Image) -> Image.Image:
        """
        階段 1: 使用 BiRefNet 進行主要分割

        Args:
            image: 輸入圖片 (RGB 或 RGBA)

        Returns:
            分割後的 RGBA 圖片
        """
        if self._session is None:
            raise RuntimeError("Model not loaded")

        # 準備輸入
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        input_data = buffer.getvalue()

        # 計算 alpha matting 參數
        enable_matting, fg_threshold, bg_threshold = self._calculate_matting_params()

        logger.debug(
            "Primary segmentation: matting=%s, fg=%d, bg=%d",
            enable_matting,
            fg_threshold,
            bg_threshold,
        )

        # 執行分割
        remove_func = cast(RemoveFunc, remove)
        output_data = remove_func(
            input_data,
            session=self._session,
            alpha_matting=enable_matting,
            alpha_matting_foreground_threshold=fg_threshold,
            alpha_matting_background_threshold=bg_threshold,
            alpha_matting_erode_size=int(5 + self.strength * 10),  # 5-15
        )

        # 轉回 PIL Image
        return Image.open(io.BytesIO(output_data)).convert("RGBA")

    def _apply_color_filter(self, image: Image.Image) -> Image.Image:
        """
        階段 2: 可選的色彩過濾（針對純色背景）

        針對純黑、純白、純綠背景進行額外的邊緣優化

        Args:
            image: RGBA 圖片

        Returns:
            過濾後的 RGBA 圖片
        """
        if not self.color_filter.enabled or self.color_filter.color == ColorFilter.NONE:
            return image

        logger.debug("Applying color filter: %s", self.color_filter.color.value)

        # 轉為 numpy 處理
        rgba = np.array(image)
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        # 轉換到 HSV 和 LAB 色彩空間以更準確判斷顏色
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        # 建立色彩遮罩
        if self.color_filter.color == ColorFilter.GREEN:
            # 綠幕檢測：使用 HSV 色相
            lower = np.array(
                [
                    self.color_filter.hue_range[0],
                    self.color_filter.saturation_min,
                    40,
                ]
            )
            upper = np.array([self.color_filter.hue_range[1], 255, 255])
            color_mask = cv2.inRange(hsv, lower, upper)

        elif self.color_filter.color == ColorFilter.BLACK:
            # 黑色檢測：使用 LAB 的 L 通道（亮度）
            l_channel = lab[:, :, 0]
            color_mask = (l_channel < 30).astype(np.uint8) * 255

        elif self.color_filter.color == ColorFilter.WHITE:
            # 白色檢測：使用 LAB 的 L 通道
            l_channel = lab[:, :, 0]
            color_mask = (l_channel > 225).astype(np.uint8) * 255

        else:
            return image

        # 優化遮罩：形態學操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # 腐蝕遮罩邊緣以避免顏色邊
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        color_mask = cv2.erode(color_mask, erode_kernel, iterations=1)

        # 羽化遮罩邊緣
        color_mask = cv2.GaussianBlur(color_mask, (5, 5), 0)

        # 反轉遮罩（0 = 背景要移除，255 = 前景保留）
        foreground_mask = 255 - color_mask

        # 與現有 alpha 合併（取最小值 = 交集）
        new_alpha = np.minimum(alpha, foreground_mask)

        # 邊緣去污染（despill）- 針對綠幕
        if self.color_filter.color == ColorFilter.GREEN:
            rgb_float = rgb.astype(np.float32)
            r, g, b = rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]

            # 計算紅藍平均
            rb_avg = (r + b) / 2

            # 綠色過量部分
            green_excess = np.maximum(g - rb_avg, 0)

            # 降低綠色（強度由 edge_refine_strength 控制）
            new_g = g - (green_excess * self.color_filter.edge_refine_strength)
            rgb_float[:, :, 1] = np.clip(new_g, 0, 255)

            rgb = rgb_float.astype(np.uint8)

        # 組合結果
        result_rgba = np.dstack([rgb, new_alpha])
        return Image.fromarray(result_rgba, "RGBA")

    def _apply_edge_defringing(self, image: Image.Image) -> Image.Image:
        """
        階段 3: 邊緣去污染（Defringing）

        移除半透明邊緣的顏色滲透，特別是白邊、灰邊

        Args:
            image: RGBA 圖片

        Returns:
            處理後的 RGBA 圖片
        """
        # 只在中高強度時啟用
        if self.strength < 0.5:  # noqa: PLR2004
            return image

        logger.debug("Applying edge defringing")

        rgba = np.array(image, dtype=np.float32)
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3] / 255.0  # 歸一化到 0-1

        # 找出邊緣區域（半透明像素）
        edge_mask = (alpha > 0.01) & (alpha < 0.99)

        if not np.any(edge_mask):
            return image

        # 對邊緣像素進行顏色調整
        # 計算顏色平均值（用於判斷是否有色偏）
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # 計算色彩不平衡度
        color_mean = (r + g + b) / 3
        r_diff = r - color_mean
        g_diff = g - color_mean
        b_diff = b - color_mean

        # 在邊緣區域減少色偏（強度基於 strength）
        defringe_strength = (self.strength - 0.5) * 2  # 0.5->0, 1.0->1.0
        r[edge_mask] -= r_diff[edge_mask] * defringe_strength * 0.5
        g[edge_mask] -= g_diff[edge_mask] * defringe_strength * 0.5
        b[edge_mask] -= b_diff[edge_mask] * defringe_strength * 0.5

        # 限制範圍
        rgb[:, :, 0] = np.clip(r, 0, 255)
        rgb[:, :, 1] = np.clip(g, 0, 255)
        rgb[:, :, 2] = np.clip(b, 0, 255)

        # 組合結果
        rgba[:, :, :3] = rgb
        return Image.fromarray(rgba.astype(np.uint8), "RGBA")

    def process(self, input_path: Path, output_path: Path) -> bool:
        """
        處理單張圖片

        完整的多階段處理流程：
        1. BiRefNet 主分割 + Alpha Matting
        2. 可選色彩過濾（純色背景）
        3. 邊緣去污染

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
            logger.debug("Processing: %s (%dx%d)", input_path.name, *original.size)

            # 階段 1: 主要分割
            result = self._apply_primary_segmentation(original)

            # 階段 2: 色彩過濾（如果啟用）
            result = self._apply_color_filter(result)

            # 階段 3: 邊緣去污染
            result = self._apply_edge_defringing(result)

            # 儲存結果
            result.save(output_path, "PNG", optimize=True)
            logger.debug("Saved: %s", output_path.name)

        except Exception:
            logger.exception("Unified backend failed: %s", input_path.name)
            return False
        else:
            return True

    @classmethod
    def get_available_models(cls) -> list[str]:
        """
        取得可用模型列表

        統一後端只使用一個最佳模型，但返回選項以符合介面
        """
        return ["auto"]

    @classmethod
    def get_model_description(cls) -> str:
        """取得模型說明"""
        return """
  統一背景移除方案：

    auto - 自動配置（推薦）
           內部使用 BiRefNet-massive（SOTA 模型）
           自動啟用 Alpha Matting 邊緣精細化
           可選純色背景過濾
           可調強度：0.1-1.0

  推薦設定：
    • 一般圖片：強度 0.5-0.7
    • 複雜邊緣（頭髮、毛髮）：強度 0.6-0.8
    • 純色背景：啟用色彩過濾 + 強度 0.7-0.9
"""
