"""
批次比對後端

提供背景移除參數批次測試功能，整合到統一的後端架構中
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

from src.backends.registry import BackendRegistry
from src.core.interfaces import BaseBackend
from src.features.batch_compare.engine import BatchCompareEngine
from src.features.batch_compare.report import ReportGenerator


logger = logging.getLogger(__name__)


@BackendRegistry.register("batch-compare")
class BatchCompareBackend(BaseBackend):
    """
    批次比對後端

    執行多組配置的批次測試，生成可視化比對報告
    """

    name: ClassVar[str] = "batch-compare"
    description: ClassVar[str] = "批次比對 - 測試多組配置並生成比對報告"

    def __init__(
        self,
        model: str = "default",
        strength: float = 0.8,
        auto_open: bool = True,
    ) -> None:
        """
        初始化後端

        Args:
            model: 模型選項（default / quick）
            strength: 未使用，保持介面一致性
            auto_open: 是否自動打開報告
        """
        super().__init__(strength=strength)
        self.model = model
        self.auto_open = auto_open
        self._engine: BatchCompareEngine | None = None
        self._input_dir: Path | None = None

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模型列表"""
        return ["default", "quick"]

    @classmethod
    def get_model_description(cls) -> str:
        """取得模型說明"""
        return "default: 完整測試所有配置（5 組）\nquick: 快速測試主要配置（3 組）"

    def load_model(self) -> None:
        """初始化比對引擎"""
        from src.features.batch_compare.engine import (  # noqa: PLC0415
            DEFAULT_TEST_CONFIGS,
        )

        if self.model == "quick":
            # 快速模式只測試前 3 個配置
            configs = DEFAULT_TEST_CONFIGS[:3]
        else:
            configs = DEFAULT_TEST_CONFIGS

        self._engine = BatchCompareEngine(configs=configs)
        logger.info("批次比對引擎已初始化 (模式: %s)", self.model)

    def process(self, input_path: Path, output_path: Path) -> bool:
        """
        執行批次比對

        注意：此後端的 process 方法與一般後端不同：
        - input_path: 輸入圖片目錄（而非單張圖片）
        - output_path: 輸出目錄/compare.html

        Args:
            input_path: 輸入圖片目錄
            output_path: 輸出路徑

        Returns:
            處理是否成功
        """
        try:
            self.ensure_model_loaded()

            # 確保輸入是目錄
            input_dir = input_path.parent if input_path.is_file() else input_path

            # 確保輸出是目錄
            output_dir = output_path.parent if output_path.suffix else output_path

            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("開始批次比對測試...")
            logger.info("輸入目錄: %s", input_dir)
            logger.info("輸出目錄: %s", output_dir)

            # 執行測試
            if self._engine is None:
                logger.error("引擎未初始化")
                return False

            result = self._engine.run(input_dir, output_dir)

            # 生成報告
            generator = ReportGenerator(input_dir, output_dir)
            report_path = generator.generate(result)

            logger.info("測試完成！最佳配置: %s", result.best_config)
            logger.info("報告位置: %s", report_path)

            # 自動打開報告
            if self.auto_open:
                self._open_report(report_path)

        except Exception:
            logger.exception("批次比對失敗")
            return False
        else:
            return True

    def _open_report(self, report_path: Path) -> None:  # noqa: PLR6301
        """在瀏覽器中打開報告"""
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(report_path)], check=False)  # noqa: S603, S607
            elif sys.platform == "win32":
                subprocess.run(  # noqa: S602, S603
                    ["start", str(report_path)],  # noqa: S607
                    shell=True,
                    check=False,
                )
            else:
                subprocess.run(["xdg-open", str(report_path)], check=False)  # noqa: S603, S607
        except Exception as e:
            logger.warning("無法自動打開報告: %s", e)

    def run_comparison(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """
        執行比對並返回詳細結果（供程式化調用）

        Args:
            input_dir: 輸入圖片目錄
            output_dir: 輸出目錄
            progress_callback: 進度回調

        Returns:
            測試結果字典
        """
        self.ensure_model_loaded()

        if self._engine is None:
            raise RuntimeError("引擎未初始化")

        # 設置進度回調
        self._engine.progress_callback = progress_callback

        # 執行測試
        result = self._engine.run(input_dir, output_dir)

        # 生成報告
        generator = ReportGenerator(input_dir, output_dir)
        report_path = generator.generate(result)

        return {
            "best_config": result.best_config,
            "test_images": result.test_images,
            "report_path": str(report_path),
            "configs": {
                name: {
                    "description": cfg.description,
                    "average_score": cfg.average_score,
                }
                for name, cfg in result.configs.items()
            },
        }
