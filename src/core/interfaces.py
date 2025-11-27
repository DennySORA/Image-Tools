"""
介面定義模組

定義系統中的抽象介面，遵循介面隔離原則 (ISP) 和依賴反轉原則 (DIP)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class BackendProtocol(Protocol):
    """
    後端協議 - 定義所有背景移除後端必須實作的介面

    遵循里氏替換原則 (LSP)：所有實作此協議的類別都可以互相替換

    Attributes:
        name: 後端名稱
        description: 後端描述
        strength: 去背強度 (0.1-1.0)
    """

    name: str
    description: str
    strength: float

    def load_model(self) -> None:
        """載入模型"""
        ...

    def process(self, input_path: Path, output_path: Path) -> bool:
        """
        處理單張圖片

        Args:
            input_path: 輸入圖片路徑
            output_path: 輸出圖片路徑

        Returns:
            處理是否成功
        """
        ...

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模型列表"""
        ...

    @classmethod
    def get_model_description(cls) -> str:
        """取得模型說明"""
        ...


class BaseBackend(ABC):
    """
    後端抽象基類

    提供共用功能，遵循單一職責原則 (SRP)
    """

    name: str = ""
    description: str = ""

    def __init__(self, strength: float = 0.5):
        """
        初始化後端

        Args:
            strength: 去背強度 (0.1-1.0)
        """
        self.strength = max(0.1, min(1.0, strength))
        self._model_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """載入模型 - 子類別必須實作"""
        pass

    @abstractmethod
    def process(self, input_path: Path, output_path: Path) -> bool:
        """處理單張圖片 - 子類別必須實作"""
        pass

    @classmethod
    @abstractmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模型列表 - 子類別必須實作"""
        pass

    @classmethod
    @abstractmethod
    def get_model_description(cls) -> str:
        """取得模型說明 - 子類別必須實作"""
        pass

    def ensure_model_loaded(self) -> None:
        """確保模型已載入"""
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
