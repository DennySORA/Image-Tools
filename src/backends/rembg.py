"""
Rembg 背景移除後端

使用 rembg 套件進行背景移除，支援多種模型：
- BiRefNet 系列 (效果最好)
- ISNet 系列
- U2Net 系列
"""

from pathlib import Path
from typing import Optional, ClassVar

from rembg import remove, new_session

from src.core.interfaces import BaseBackend
from .registry import BackendRegistry


# 可用的模型列表
AVAILABLE_MODELS: tuple[str, ...] = (
    # BiRefNet 系列 (效果最好)
    'birefnet-general',
    'birefnet-general-lite',
    'birefnet-portrait',
    'birefnet-massive',
    # ISNet 系列
    'isnet-general-use',
    'isnet-anime',
    # U2Net 系列
    'u2net',
    'u2netp',
    'u2net_human_seg',
    'silueta',
)

DEFAULT_MODEL: str = 'birefnet-general'


@BackendRegistry.register("rembg")
class RembgBackend(BaseBackend):
    """
    Rembg 背景移除後端

    支援多種模型，使用 BiRefNet 效果最好
    """

    name: ClassVar[str] = "rembg"
    description: ClassVar[str] = "Rembg - 支援多種模型 (BiRefNet, ISNet, U2Net)"

    def __init__(self, model: str = DEFAULT_MODEL, strength: float = 0.5):
        """
        初始化 Rembg 後端

        Args:
            model: 使用的模型名稱
            strength: 去背強度 (0.1-1.0)

        Raises:
            ValueError: 當模型不支援時
        """
        super().__init__(strength=strength)

        if model not in AVAILABLE_MODELS:
            raise ValueError(f"不支援的模型: {model}，可用模型: {AVAILABLE_MODELS}")

        self.model = model
        self._session: Optional[object] = None

    def load_model(self) -> None:
        """載入模型"""
        print(f"[Rembg] 載入模型: {self.model}")
        print(f"[Rembg] 去背強度: {self.strength}")
        self._session = new_session(self.model)
        print(f"[Rembg] 模型載入完成")

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

            # 根據強度調整參數
            alpha_matting = self.strength >= 0.5
            fg_threshold = int(255 - (self.strength * 50))
            bg_threshold = int(self.strength * 30)

            output_data = remove(
                input_data,
                session=self._session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=fg_threshold,
                alpha_matting_background_threshold=bg_threshold,
            )

            with open(output_path, 'wb') as f:
                f.write(output_data)

            return True

        except Exception as e:
            print(f"[Rembg] 處理失敗 {input_path.name}: {e}")
            return False

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模型列表"""
        return list(AVAILABLE_MODELS)

    @classmethod
    def get_model_description(cls) -> str:
        """取得模型說明"""
        return """
  BiRefNet 系列 (推薦):
    birefnet-general      - 通用場景，效果最好
    birefnet-general-lite - 輕量版，速度較快
    birefnet-portrait     - 人像專用
    birefnet-massive      - 大型資料集訓練

  ISNet 系列:
    isnet-general-use     - 通用場景
    isnet-anime           - 動漫角色

  U2Net 系列:
    u2net                 - 經典模型
    u2netp                - 輕量版
    u2net_human_seg       - 人體分割
    silueta               - u2net 壓縮版
"""
