"""
資料模型模組

定義系統中使用的資料結構，遵循單一職責原則 (SRP)
使用 dataclass 確保資料的不可變性和清晰性
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    """
    模型資訊

    Attributes:
        name: 模型名稱
        description: 模型描述
    """
    name: str
    description: str


@dataclass(frozen=True)
class BackendInfo:
    """
    後端資訊

    Attributes:
        name: 後端名稱
        description: 後端描述
        models: 可用模型列表
    """
    name: str
    description: str
    models: tuple[ModelInfo, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProcessConfig:
    """
    處理設定

    封裝所有處理相關的設定，遵循單一職責原則

    Attributes:
        input_folder: 輸入資料夾路徑
        backend_name: 使用的後端名稱
        model: 使用的模型名稱
        strength: 去背強度 (0.1-1.0)
        output_folder: 輸出資料夾路徑 (預設為 input_folder/output)
    """
    input_folder: Path
    backend_name: str
    model: str
    strength: float
    output_folder: Path = field(default=None)

    def __post_init__(self):
        # frozen=True 時需要使用 object.__setattr__
        if self.output_folder is None:
            object.__setattr__(self, 'output_folder', self.input_folder / 'output')


@dataclass(frozen=True)
class ProcessResult:
    """
    處理結果

    封裝處理完成後的結果資訊

    Attributes:
        total: 總圖片數
        success: 成功數
        failed: 失敗數
        output_folder: 輸出資料夾路徑
    """
    total: int
    success: int
    failed: int
    output_folder: Path

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success / self.total if self.total > 0 else 0.0

    @property
    def is_complete_success(self) -> bool:
        """是否全部成功"""
        return self.failed == 0


@dataclass
class ImageFile:
    """
    圖片檔案資訊

    Attributes:
        path: 檔案路徑
        name: 檔案名稱
    """
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()


# 支援的圖片格式
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'
})


def is_supported_image(path: Path) -> bool:
    """檢查檔案是否為支援的圖片格式"""
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
