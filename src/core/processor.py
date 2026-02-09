"""
圖片處理器模組

負責圖片的批次處理邏輯，遵循單一職責原則 (SRP)
依賴抽象介面而非具體實作，遵循依賴反轉原則 (DIP)
"""

import logging
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.data_model import (
    SUPPORTED_EXTENSIONS,
    ProcessConfig,
    ProcessResult,
    is_supported_image,
)

from .interfaces import BackendProtocol


logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    圖片處理器

    負責批次處理資料夾中的圖片，遵循單一職責原則
    支援序列和並行處理模式
    """

    def __init__(
        self,
        backend: BackendProtocol,
        progress_callback: Callable[[int, int, str], None] | None = None,
        max_workers: int = 1,
    ):
        """
        初始化處理器

        Args:
            backend: 背景移除後端
            progress_callback: 進度回調函數 (current, total, filename)
            max_workers: 並行工作執行緒數（1=序列處理，>1=並行處理）
        """
        self._backend = backend
        self._progress_callback = progress_callback or self._default_progress
        self._max_workers = max(1, max_workers)
        self._progress_lock = threading.Lock()

    @staticmethod
    def _default_progress(current: int, total: int, filename: str) -> None:
        """預設進度顯示"""
        sys.stdout.write(f"[{current}/{total}] {filename} ... ")
        sys.stdout.flush()

    def scan_images(self, folder: Path) -> list[Path]:
        """
        掃描資料夾中的圖片檔案

        Args:
            folder: 資料夾路徑

        Returns:
            圖片檔案路徑列表
        """
        return [f for f in sorted(folder.iterdir()) if is_supported_image(f)]

    def process_folder(self, config: ProcessConfig) -> ProcessResult:
        """
        處理資料夾中的所有圖片

        根據 max_workers 自動選擇序列或並行模式

        Args:
            config: 處理設定

        Returns:
            處理結果
        """
        output_folder = config.output_folder
        if output_folder is None:
            raise ValueError("Output folder is not set")

        # 確保輸出資料夾存在
        output_folder.mkdir(parents=True, exist_ok=True)

        # 掃描圖片
        image_files = self.scan_images(config.input_folder)
        total = len(image_files)

        if total == 0:
            return ProcessResult(
                total=0,
                success=0,
                failed=0,
                output_folder=output_folder,
            )

        # 載入模型
        self._backend.load_model()

        # 根據 worker 數量選擇處理模式
        if self._max_workers <= 1 or total == 1:
            success_count = self._process_sequential(image_files, output_folder, total)
        else:
            success_count = self._process_parallel(image_files, output_folder, total)

        return ProcessResult(
            total=total,
            success=success_count,
            failed=total - success_count,
            output_folder=output_folder,
        )

    def _process_sequential(
        self,
        image_files: list[Path],
        output_folder: Path,
        total: int,
    ) -> int:
        """序列處理所有圖片"""
        success_count = 0

        for i, image_path in enumerate(image_files, 1):
            output_path = output_folder / f"{image_path.stem}.png"

            self._progress_callback(i, total, image_path.name)

            if self._backend.process(image_path, output_path):
                sys.stdout.write("完成\n")
                sys.stdout.flush()
                success_count += 1
            else:
                sys.stdout.write("失敗\n")
                sys.stdout.flush()

        return success_count

    def _process_parallel(
        self,
        image_files: list[Path],
        output_folder: Path,
        total: int,
    ) -> int:
        """並行處理所有圖片"""
        success_count = 0
        completed_count = 0

        logger.info(
            "Parallel processing: %d images with %d workers",
            total,
            self._max_workers,
        )

        def _process_one(image_path: Path) -> tuple[str, bool]:
            output_path = output_folder / f"{image_path.stem}.png"
            result = self._backend.process(image_path, output_path)
            return image_path.name, result

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(_process_one, path): path for path in image_files
            }

            for future in as_completed(futures):
                filename, result = future.result()
                completed_count += 1

                with self._progress_lock:
                    self._progress_callback(completed_count, total, filename)
                    if result:
                        sys.stdout.write("完成\n")
                        sys.stdout.flush()
                        success_count += 1
                    else:
                        sys.stdout.write("失敗\n")
                        sys.stdout.flush()

        return success_count

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
