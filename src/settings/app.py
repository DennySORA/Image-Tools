"""
應用程式設定

使用 Pydantic BaseSettings 管理環境變數和配置
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    應用程式設定

    從環境變數和 .env 文件讀取設定

    Attributes:
        log_level: 日誌級別 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        cache_dir: 快取目錄（用於存儲模型等）
        max_image_size: 最大圖片尺寸（像素）
        default_backend: 預設後端名稱
        default_strength: 預設處理強度
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="REMBG_",
        case_sensitive=False,
    )

    # 日誌設定
    log_level: str = "INFO"

    # 快取設定
    cache_dir: Path = Path.home() / ".cache" / "rembg-tool"

    # 圖片處理設定
    max_image_size: int = 4096  # 最大邊長（像素）
    default_backend: str = "ultra"
    default_strength: float = 0.8

    # 模型設定
    model_cache_enabled: bool = True
    model_download_timeout: int = 300  # 秒


# 創建全局設定實例
settings = AppSettings()
