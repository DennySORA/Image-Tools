"""
Transparent Background 背景移除後端

使用 transparent-background 套件 (InSPyReNet 模型) 進行背景移除
"""

from pathlib import Path
from typing import Optional, ClassVar

from PIL import Image
from transparent_background import Remover

from src.core.interfaces import BaseBackend
from .registry import BackendRegistry


# 可用的模式
AVAILABLE_MODES: tuple[str, ...] = (
    'base',
    'fast',
    'base-nightly',
)

DEFAULT_MODE: str = 'base'


@BackendRegistry.register("transparent-background")
class TransparentBgBackend(BaseBackend):
    """
    Transparent Background 背景移除後端

    使用 InSPyReNet 模型，效果穩定
    """

    name: ClassVar[str] = "transparent-background"
    description: ClassVar[str] = "Transparent Background - 使用 InSPyReNet 模型"

    def __init__(self, mode: str = DEFAULT_MODE, strength: float = 0.5):
        """
        初始化 Transparent Background 後端

        Args:
            mode: 使用的模式 (base, fast, base-nightly)
            strength: 去背強度 (0.1-1.0)

        Raises:
            ValueError: 當模式不支援時
        """
        super().__init__(strength=strength)

        if mode not in AVAILABLE_MODES:
            raise ValueError(f"不支援的模式: {mode}，可用模式: {AVAILABLE_MODES}")

        self.mode = mode
        self._remover: Optional[Remover] = None

    def load_model(self) -> None:
        """載入模型"""
        print(f"[TransparentBg] 載入模型: mode={self.mode}")
        print(f"[TransparentBg] 去背強度: {self.strength}")
        self._remover = Remover(mode=self.mode)
        print(f"[TransparentBg] 模型載入完成")

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
            img = Image.open(input_path).convert('RGB')

            # 根據強度設定閾值
            threshold = 1.0 - (self.strength * 0.5)

            out = self._remover.process(img, type='rgba', threshold=threshold)
            out.save(output_path)

            return True

        except Exception as e:
            print(f"[TransparentBg] 處理失敗 {input_path.name}: {e}")
            return False

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模式列表"""
        return list(AVAILABLE_MODES)

    @classmethod
    def get_model_description(cls) -> str:
        """取得模式說明"""
        return """
  可用模式:
    base         - 預設模式，平衡速度與品質
    fast         - 快速模式，速度較快但品質稍低
    base-nightly - nightly 版本，可能有最新改進
"""
