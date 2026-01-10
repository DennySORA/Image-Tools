"""
BackgroundRemover 背景移除後端

使用 backgroundremover 套件進行背景移除，支援 Alpha Matting
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, cast

# Monkeypatch moviepy to fix compatibility with backgroundremover
try:
    import moviepy
    import moviepy.editor

    if not hasattr(moviepy, "VideoFileClip"):
        moviepy.VideoFileClip = moviepy.editor.VideoFileClip
except ImportError:
    pass

from backgroundremover import bg as background_bg  # type: ignore[import-untyped]

from src.core.interfaces import BaseBackend

from .registry import BackendRegistry


# 可用的模型列表
AVAILABLE_MODELS: tuple[str, ...] = (
    "u2net",
    "u2net_human_seg",
    "u2netp",
)

DEFAULT_MODEL: str = "u2net"

RemoveFunc = Callable[..., bytes]

logger = logging.getLogger(__name__)


@BackendRegistry.register("backgroundremover")
class BackgroundRemoverBackend(BaseBackend):
    """
    BackgroundRemover 背景移除後端

    支援 Alpha Matting 邊緣優化
    """

    name: ClassVar[str] = "backgroundremover"
    description: ClassVar[str] = "BackgroundRemover - 支援 Alpha Matting 邊緣優化"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        strength: float = 0.5,
        alpha_matting: bool = True,
    ):
        """
        初始化 BackgroundRemover 後端

        Args:
            model: 使用的模型名稱
            strength: 去背強度 (0.1-1.0)
            alpha_matting: 是否啟用 alpha matting

        Raises:
            ValueError: 當模型不支援時
        """
        super().__init__(strength=strength)

        if model not in AVAILABLE_MODELS:
            raise ValueError(f"不支援的模型: {model}，可用模型: {AVAILABLE_MODELS}")

        self.model = model
        self.alpha_matting = alpha_matting

        # 根據強度調整參數
        self.foreground_threshold = int(255 - (self.strength * 30))
        self.background_threshold = int(self.strength * 20)
        self.erode_size = max(1, min(25, int(10 * self.strength * 2)))

        self._remove_func: RemoveFunc | None = None

    def load_model(self) -> None:
        """載入模型"""
        logger.info("BackgroundRemover model: %s", self.model)
        logger.info("BackgroundRemover strength: %s", self.strength)
        logger.info("BackgroundRemover alpha matting: %s", self.alpha_matting)

        if self.alpha_matting:
            logger.info(
                "BackgroundRemover foreground threshold: %s", self.foreground_threshold
            )
            logger.info(
                "BackgroundRemover background threshold: %s", self.background_threshold
            )
            logger.info("BackgroundRemover erode size: %s", self.erode_size)

        self._remove_func = cast(RemoveFunc, background_bg.remove)

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

        remove_func = self._remove_func
        if remove_func is None:
            logger.error("BackgroundRemover remove function not loaded")
            return False

        try:
            with open(input_path, "rb") as f:
                input_data = f.read()

            output_data = remove_func(
                input_data,
                model_name=self.model,
                alpha_matting=self.alpha_matting,
                alpha_matting_foreground_threshold=self.foreground_threshold,
                alpha_matting_background_threshold=self.background_threshold,
                alpha_matting_erode_structure_size=self.erode_size,
            )

            with open(output_path, "wb") as f:
                f.write(output_data)
        except Exception:
            logger.exception("BackgroundRemover failed: %s", input_path.name)
            return False
        else:
            return True

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模型列表"""
        return list(AVAILABLE_MODELS)

    @classmethod
    def get_model_description(cls) -> str:
        """取得模型說明"""
        return """
  可用模型:
    u2net           - 通用模型 (預設)
    u2net_human_seg - 人像專用，精度最高
    u2netp          - 快速版本，精度較低

  Alpha Matting 參數會根據去背強度自動調整
"""
