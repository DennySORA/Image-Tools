"""
綠幕專用混合方案後端

結合三層處理技術：
1. 色度鍵預處理：移除純綠色背景
2. AI 精細化：使用 rembg 優化邊緣
3. Despill 後處理：移除綠色溢出

專為綠幕背景設計，效果優於單純 AI 去背
"""

import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
from PIL import Image
from rembg import new_session, remove  # type: ignore[import-untyped]

from src.core.interfaces import BaseBackend
from src.postprocess.green_screen import GreenScreenConfig, GreenScreenProcessor

from .registry import BackendRegistry


# 可用的處理模式
AVAILABLE_MODES: tuple[str, ...] = (
    "hybrid",  # 混合模式：色度鍵 + AI + Despill (推薦)
    "chroma-only",  # 純色度鍵模式：速度最快
    "ai-enhanced",  # AI 增強模式：色度鍵 + AI
)

DEFAULT_MODE: str = "hybrid"

RemoveFunc = Callable[..., bytes]
SessionFactory = Callable[[str], object]

logger = logging.getLogger(__name__)


@BackendRegistry.register("greenscreen")
class GreenScreenBackend(BaseBackend):
    """
    綠幕專用混合方案後端

    專為綠幕背景設計，結合色度鍵和 AI 技術
    """

    name: ClassVar[str] = "greenscreen"
    description: ClassVar[str] = "綠幕專用 - 混合方案 (色度鍵 + AI + Despill)"

    def __init__(
        self,
        model: str = DEFAULT_MODE,
        strength: float = 0.7,
        hue_range: tuple[int, int] = (35, 85),
        saturation_min: int = 40,
    ):
        """
        初始化綠幕後端

        Args:
            model: 處理模式 (hybrid, chroma-only, ai-enhanced)
            strength: 處理強度，影響 despill 和邊緣處理
            hue_range: 綠色色相範圍 (HSV)
            saturation_min: 最低飽和度閾值
        """
        super().__init__(strength=strength)

        if model not in AVAILABLE_MODES:
            raise ValueError(f"不支援的模式: {model}，可用模式: {AVAILABLE_MODES}")

        self.mode = model
        self.hue_range = hue_range
        self.saturation_min = saturation_min

        # 綠幕處理器設定
        self._gs_config = GreenScreenConfig(
            hue_min=hue_range[0],
            hue_max=hue_range[1],
            saturation_min=saturation_min,
            despill_strength=strength,
            edge_blur=3,
            erode_size=2,
            feather_amount=2,
        )
        self._gs_processor: GreenScreenProcessor | None = None
        self._ai_session: object | None = None

    def load_model(self) -> None:
        """載入模型"""
        logger.info("GreenScreen mode: %s", self.mode)
        logger.info(
            "GreenScreen hue range: H=%s, S>=%s",
            self.hue_range,
            self.saturation_min,
        )
        logger.info("GreenScreen strength: %s", self.strength)

        # 初始化綠幕處理器
        self._gs_processor = GreenScreenProcessor(self._gs_config)

        # 如果需要 AI，載入 rembg 模型
        if self.mode in ("hybrid", "ai-enhanced"):
            logger.info("GreenScreen AI model: isnet-anime")
            session_factory = cast(SessionFactory, new_session)
            self._ai_session = session_factory("isnet-anime")

        logger.info("GreenScreen models loaded")

    def _apply_chroma_key(self, image: Image.Image) -> Image.Image:
        """
        應用色度鍵處理

        Args:
            image: 輸入圖片

        Returns:
            處理後的 RGBA 圖片
        """
        processor = self._gs_processor
        if processor is None:
            raise RuntimeError("GreenScreen processor not initialized")
        return processor.process_image(image)

    def _apply_ai_refinement(self, image: Image.Image) -> Image.Image:
        """
        應用 AI 精細化處理

        Args:
            image: 輸入圖片 (可以是 RGB 或 RGBA)

        Returns:
            AI 處理後的 RGBA 圖片
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        input_data = buffer.getvalue()

        # AI 去背
        if self._ai_session is None:
            raise RuntimeError("GreenScreen AI session not initialized")

        remove_func = cast(RemoveFunc, remove)
        output_data = remove_func(
            input_data,
            session=self._ai_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=20,
        )

        # 轉回 PIL Image
        return Image.open(io.BytesIO(output_data)).convert("RGBA")

    def _apply_despill(self, image: Image.Image) -> Image.Image:
        """
        應用 despill 後處理

        移除邊緣的綠色溢出

        Args:
            image: RGBA 圖片

        Returns:
            處理後的圖片
        """
        rgba = np.array(image)

        # 只處理半透明和不透明的像素
        alpha = rgba[:, :, 3]
        visible_mask = alpha > 0

        if not np.any(visible_mask):
            return image

        # RGB 通道
        r = rgba[:, :, 0].astype(np.float32)
        g = rgba[:, :, 1].astype(np.float32)
        b = rgba[:, :, 2].astype(np.float32)

        # 計算紅藍平均
        rb_avg = (r + b) / 2

        # 綠色過量
        green_excess = np.maximum(g - rb_avg, 0)

        # 降低綠色
        new_g = g - (green_excess * self.strength)
        rgba[:, :, 1] = np.clip(new_g, 0, 255).astype(np.uint8)

        return Image.fromarray(rgba, "RGBA")

    def _merge_alpha_channels(
        self, chroma_image: Image.Image, ai_image: Image.Image
    ) -> Image.Image:
        """
        合併色度鍵和 AI 的 alpha 通道

        使用兩者的交集來確保最乾淨的邊緣

        Args:
            chroma_image: 色度鍵處理後的圖片
            ai_image: AI 處理後的圖片

        Returns:
            合併後的圖片
        """
        chroma_rgba = np.array(chroma_image)
        ai_rgba = np.array(ai_image)

        # 取 alpha 交集 (最小值)
        merged_alpha = np.minimum(chroma_rgba[:, :, 3], ai_rgba[:, :, 3])

        # 使用 AI 的 RGB (通常邊緣更好)
        result = ai_rgba.copy()
        result[:, :, 3] = merged_alpha

        return Image.fromarray(result, "RGBA")

    def process(self, input_path: Path, output_path: Path) -> bool:
        """
        處理單張圖片

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

            if self.mode == "chroma-only":
                # 純色度鍵模式
                result = self._apply_chroma_key(original)
                result = self._apply_despill(result)
            elif self.mode == "ai-enhanced":
                # 色度鍵 + AI
                chroma_result = self._apply_chroma_key(original)
                ai_result = self._apply_ai_refinement(original)
                result = self._merge_alpha_channels(chroma_result, ai_result)
            else:  # hybrid (預設)
                # 完整混合模式：色度鍵 + AI + Despill
                chroma_result = self._apply_chroma_key(original)
                ai_result = self._apply_ai_refinement(original)
                merged = self._merge_alpha_channels(chroma_result, ai_result)
                result = self._apply_despill(merged)

            # 儲存結果
            result.save(output_path, "PNG")
        except Exception:
            logger.exception("GreenScreen failed: %s", input_path.name)
            return False
        else:
            return True

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模式列表"""
        return list(AVAILABLE_MODES)

    @classmethod
    def get_model_description(cls) -> str:
        """取得模式說明"""
        return """
  綠幕專用模式 (推薦用於綠幕背景):

    hybrid        - 混合模式 (推薦)
                    色度鍵 + AI 精細化 + Despill
                    效果最好，速度適中

    chroma-only   - 純色度鍵模式
                    只使用色度鍵技術
                    速度最快，適合純色綠幕

    ai-enhanced   - AI 增強模式
                    色度鍵 + AI 精細化
                    無 despill，保留原始色彩
"""
