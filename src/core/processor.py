"""
圖片處理器模組

負責圖片的批次處理邏輯，遵循單一職責原則 (SRP)
依賴抽象介面而非具體實作，遵循依賴反轉原則 (DIP)
"""

from pathlib import Path
from typing import Callable, Optional

from .interfaces import BackendProtocol
from .models import ProcessConfig, ProcessResult, SUPPORTED_EXTENSIONS, is_supported_image


class ImageProcessor:
    """
    圖片處理器

    負責批次處理資料夾中的圖片，遵循單一職責原則
    """

    def __init__(
        self,
        backend: BackendProtocol,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        初始化處理器

        Args:
            backend: 背景移除後端
            progress_callback: 進度回調函數 (current, total, filename)
        """
        self._backend = backend
        self._progress_callback = progress_callback or self._default_progress

    @staticmethod
    def _default_progress(current: int, total: int, filename: str) -> None:
        """預設進度顯示"""
        print(f"[{current}/{total}] {filename}", end=" ... ", flush=True)

    def scan_images(self, folder: Path) -> list[Path]:
        """
        掃描資料夾中的圖片檔案

        Args:
            folder: 資料夾路徑

        Returns:
            圖片檔案路徑列表
        """
        return [
            f for f in sorted(folder.iterdir())
            if is_supported_image(f)
        ]

    def process_folder(self, config: ProcessConfig) -> ProcessResult:
        """
        處理資料夾中的所有圖片

        Args:
            config: 處理設定

        Returns:
            處理結果
        """
        # 確保輸出資料夾存在
        config.output_folder.mkdir(parents=True, exist_ok=True)

        # 掃描圖片
        image_files = self.scan_images(config.input_folder)
        total = len(image_files)

        if total == 0:
            return ProcessResult(
                total=0,
                success=0,
                failed=0,
                output_folder=config.output_folder,
            )

        # 載入模型
        self._backend.load_model()

        # 處理每張圖片
        success_count = 0

        for i, image_path in enumerate(image_files, 1):
            output_path = config.output_folder / f"{image_path.stem}.png"

            self._progress_callback(i, total, image_path.name)

            if self._backend.process(image_path, output_path):
                print("完成")
                success_count += 1
            else:
                print("失敗")

        return ProcessResult(
            total=total,
            success=success_count,
            failed=total - success_count,
            output_folder=config.output_folder,
        )

    def process_single(self, input_path: Path, output_path: Path) -> bool:
        """
        處理單張圖片

        Args:
            input_path: 輸入圖片路徑
            output_path: 輸出圖片路徑

        Returns:
            處理是否成功
        """
        self._backend.ensure_model_loaded()
        return self._backend.process(input_path, output_path)


def get_supported_extensions() -> frozenset[str]:
    """取得支援的圖片格式"""
    return SUPPORTED_EXTENSIONS
