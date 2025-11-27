"""
BackgroundRemover 背景移除後端

使用 backgroundremover 套件進行背景移除，支援 Alpha Matting
"""

from pathlib import Path
from typing import Optional, Callable, ClassVar

from src.core.interfaces import BaseBackend
from .registry import BackendRegistry


# 可用的模型列表
AVAILABLE_MODELS: tuple[str, ...] = (
    'u2net',
    'u2net_human_seg',
    'u2netp',
)

DEFAULT_MODEL: str = 'u2net'


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

        self._remove_func: Optional[Callable] = None

    def load_model(self) -> None:
        """載入模型"""
        print(f"[BackgroundRemover] 使用模型: {self.model}")
        print(f"[BackgroundRemover] 去背強度: {self.strength}")
        print(f"[BackgroundRemover] Alpha Matting: {self.alpha_matting}")

        if self.alpha_matting:
            print(f"[BackgroundRemover] 前景閾值: {self.foreground_threshold}")
            print(f"[BackgroundRemover] 背景閾值: {self.background_threshold}")
            print(f"[BackgroundRemover] 侵蝕大小: {self.erode_size}")

        # 延遲載入以避免 import 錯誤
        from backgroundremover.bg import remove
        self._remove_func = remove

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
            with open(input_path, 'rb') as f:
                input_data = f.read()

            output_data = self._remove_func(
                input_data,
                model_name=self.model,
                alpha_matting=self.alpha_matting,
                alpha_matting_foreground_threshold=self.foreground_threshold,
                alpha_matting_background_threshold=self.background_threshold,
                alpha_matting_erode_structure_size=self.erode_size,
            )

            with open(output_path, 'wb') as f:
                f.write(output_data)

            return True

        except Exception as e:
            print(f"[BackgroundRemover] 處理失敗 {input_path.name}: {e}")
            return False

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
