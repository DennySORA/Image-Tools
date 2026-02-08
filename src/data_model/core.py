"""
核心資料模型

使用 Pydantic 進行資料驗證和序列化，確保資料完整性
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelInfo(BaseModel):
    """
    模型資訊

    Attributes:
        name: 模型名稱
        description: 模型描述
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str


class BackendInfo(BaseModel):
    """
    後端資訊

    Attributes:
        name: 後端名稱
        description: 後端描述
        models: 可用模型列表
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    models: tuple[ModelInfo, ...] = Field(default_factory=tuple)


class ProcessConfig(BaseModel):
    """
    處理設定

    封裝所有處理相關的設定，遵循單一職責原則

    Attributes:
        input_folder: 輸入資料夾路徑
        backend_name: 使用的後端名稱
        model: 使用的模型名稱
        strength: 去背強度 (0.1-1.0)
        output_folder: 輸出資料夾路徑
        extra_config: 額外設定（如色彩過濾等）
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    input_folder: Path
    backend_name: str
    model: str
    strength: float = Field(ge=0.1, le=1.0)
    output_folder: Path | None = None
    extra_config: dict[str, object] = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        """Set default output folder after initialization."""
        if self.output_folder is None:
            # Use object.__setattr__ since model is frozen
            object.__setattr__(self, "output_folder", self.input_folder / "output")


class ProcessResult(BaseModel):
    """
    處理結果

    封裝處理完成後的結果資訊

    Attributes:
        total: 總圖片數
        success: 成功數
        failed: 失敗數
        output_folder: 輸出資料夾路徑
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    total: int = Field(ge=0)
    success: int = Field(ge=0)
    failed: int = Field(ge=0)
    output_folder: Path

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success / self.total if self.total > 0 else 0.0

    @property
    def is_complete_success(self) -> bool:
        """是否全部成功"""
        return self.failed == 0


class ImageFile(BaseModel):
    """
    圖片檔案資訊

    Attributes:
        path: 檔案路徑
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path

    @property
    def name(self) -> str:
        """檔案名稱"""
        return self.path.name

    @property
    def stem(self) -> str:
        """檔案名（不含副檔名）"""
        return self.path.stem

    @property
    def suffix(self) -> str:
        """副檔名（小寫）"""
        return self.path.suffix.lower()


# 支援的圖片格式
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
)


def is_supported_image(path: Path) -> bool:
    """
    檢查檔案是否為支援的圖片格式

    Args:
        path: 檔案路徑

    Returns:
        是否為支援的圖片格式
    """
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
