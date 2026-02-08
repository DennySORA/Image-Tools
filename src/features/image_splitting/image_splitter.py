"""
圖片分割後端

將透明圖片中的多個物件分割成獨立的精靈圖
"""

import logging
from pathlib import Path
from typing import ClassVar

from src.backends.registry import BackendRegistry
from src.core.interfaces import BaseBackend

from .splitter import ImageSplitter, SplitConfig


logger = logging.getLogger(__name__)


# 可用的畫布模式
CANVAS_MODES: tuple[str, ...] = (
    "max",  # 所有輸出使用相同大小 (最大裁切)
    "tight",  # 每個輸出使用最小裁切
    "original",  # 使用原始圖片大小
)

DEFAULT_CANVAS_MODE: str = "max"


@BackendRegistry.register("image-splitter")
class ImageSplitterBackend(BaseBackend):
    """
    圖片分割後端

    使用 Alpha 通道連通分量分析來分割透明圖片中的多個物件
    """

    name: ClassVar[str] = "image-splitter"
    description: ClassVar[str] = "圖片分割器 - 將精靈圖分割成獨立物件"

    def __init__(
        self,
        model: str = DEFAULT_CANVAS_MODE,
        strength: float = 0.5,
        alpha_threshold: int = 1,
        smart_threshold: bool = True,
    ):
        """
        初始化圖片分割後端

        Args:
            model: 畫布模式 (max/tight/original)
            strength: 用於控制裁切填充 (0.1-1.0)
            alpha_threshold: Alpha 閾值 (1-255)
            smart_threshold: 是否使用智能閾值檢測
        """
        super().__init__(strength=strength)

        if model not in CANVAS_MODES:
            raise ValueError(f"不支援的畫布模式: {model}，可用模式: {CANVAS_MODES}")

        self.canvas_mode = model
        self.alpha_threshold = alpha_threshold
        self.smart_threshold = smart_threshold
        self._splitter: ImageSplitter | None = None

    def load_model(self) -> None:
        """載入模型 (初始化分割器)"""
        logger.info(f"圖片分割器 - 畫布模式: {self.canvas_mode}")
        logger.info(f"Alpha 閾值: {self.alpha_threshold}")
        logger.info(f"智能閾值: {self.smart_threshold}")
        logger.info(f"裁切填充: {int(self.strength * 16)} px")

        # 計算裁切填充 (strength 0.1-1.0 對應到 2-16 像素)
        crop_padding = max(2, int(self.strength * 16))

        config = SplitConfig(
            alpha_threshold=self.alpha_threshold,
            canvas_mode=self.canvas_mode,
            crop_padding_px=crop_padding,
            smart_threshold=self.smart_threshold,
        )

        self._splitter = ImageSplitter(config)
        logger.info("圖片分割器已初始化")

    def process(self, input_path: Path, output_path: Path) -> bool:
        """
        處理單張圖片 (分割成多個精靈)

        注意: output_path 將被用作輸出目錄的基準路徑
        實際輸出檔案會是: output_path.parent / output_path.stem / {stem}_01.png, ...

        Args:
            input_path: 輸入圖片路徑
            output_path: 輸出路徑基準

        Returns:
            處理是否成功
        """
        self.ensure_model_loaded()

        if self._splitter is None:
            logger.error("圖片分割器未初始化")
            return False

        try:
            # 建立輸出目錄: output_path.parent / output_path.stem
            output_dir = output_path.parent / output_path.stem
            prefix = input_path.stem

            # 處理圖片
            count = self._splitter.process_file(input_path, output_dir, prefix)
            logger.info(f"成功分割 {input_path.name} 為 {count} 個精靈")
        except Exception:
            logger.exception(f"圖片分割失敗: {input_path.name}")
            return False
        else:
            return True

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模型列表 (畫布模式)"""
        return list(CANVAS_MODES)

    @classmethod
    def get_model_description(cls) -> str:
        """取得模型說明"""
        return """
  畫布模式:
    max      - 所有輸出使用相同大小 (最大裁切)，物件居中
    tight    - 每個輸出使用最小裁切，無額外畫布
    original - 輸出畫布等於原始圖片大小

  參數說明:
    - Alpha 閾值: 控制物件檢測的敏感度 (預設自動)
    - 強度: 控制裁切時保留的透明邊距 (0.1=2px, 1.0=16px)
    - 智能閾值: 自動嘗試不同閾值以找到最佳分割結果
"""
