"""
應用程式服務層

提供依賴注入和業務邏輯編排
"""

import logging

from src.backends.registry import BackendRegistry
from src.common import (
    AlphaConfig,
    AlphaMode,
    ColorFilter,
    ColorFilterConfig,
    ResolutionConfig,
    ResolutionMode,
)
from src.core.interfaces import BackendProtocol
from src.core.processor import ImageProcessor
from src.data_model import ProcessConfig, ProcessResult
from src.ui import ModernUI


logger = logging.getLogger(__name__)


class ApplicationService:
    """
    應用程式服務

    協調 UI、處理器和後端，實現依賴反轉原則 (DIP)
    """

    def __init__(
        self,
        ui: ModernUI | None = None,
        backend_registry: type[BackendRegistry] = BackendRegistry,
    ):
        """
        初始化應用程式服務

        Args:
            ui: 使用者介面 (可注入，預設為 ModernUI)
            backend_registry: 後端註冊表 (可注入以供測試)
        """
        self.ui = ui or ModernUI()
        self.backend_registry = backend_registry

    def run(self) -> int:
        """
        執行應用程式主循環

        Returns:
            退出碼 (0: 成功, 1: 失敗, 130: 中斷)
        """
        try:
            while True:
                # 1. 獲取使用者配置
                config = self.ui.run()
                if config is None:
                    print("\n👋 再見！")
                    return 0

                # 2. 顯示處理摘要
                self.ui.show_summary(config)

                # 3. 處理圖片
                result = self._process_images(config)

                # 4. 顯示結果
                self._display_result(result)

                # 5. 自動返回主選單
                print("🔄 返回主選單...\n")

        except KeyboardInterrupt:
            print("\n\n👋 已中斷操作，再見！")
            return 130

        except Exception:
            logger.exception("應用程式執行錯誤")
            print("\n❌ 應用程式發生錯誤，請查看日誌\n")
            return 1

    def _process_images(self, config: ProcessConfig) -> ProcessResult:
        """
        處理圖片

        Args:
            config: 處理配置

        Returns:
            處理結果
        """
        # 建立後端（應用工廠模式）
        backend = self._create_backend(config)

        # 建立處理器（注入後端依賴）
        processor = ImageProcessor(backend)

        # 處理並返回結果
        return processor.process_folder(config)

    def _create_backend(self, config: ProcessConfig) -> BackendProtocol:
        """
        建立後端實例

        Args:
            config: 處理配置

        Returns:
            後端實例
        """
        backend_kwargs = {}

        if config.backend_name == "ultra":
            backend_kwargs = self._build_ultra_kwargs(config)

        # 使用註冊表建立後端（工廠模式）
        return self.backend_registry.create(
            name=config.backend_name,
            model=config.model,
            strength=config.strength,
            **backend_kwargs,
        )

    def _build_ultra_kwargs(self, config: ProcessConfig) -> dict[str, object]:
        """
        從 extra_config 建構 Ultra 後端的完整參數

        Args:
            config: 處理配置

        Returns:
            Ultra 後端建構參數
        """
        extra = config.extra_config
        kwargs: dict[str, object] = {}

        # 色彩過濾
        color_value = str(extra.get("color_filter", "none"))
        if color_value != "none":
            kwargs["color_filter"] = ColorFilterConfig(
                enabled=True,
                color=ColorFilter(color_value),
                edge_refine_strength=config.strength,
            )

        # Trimap 精修
        if "use_trimap_refine" in extra:
            kwargs["use_trimap_refine"] = bool(extra["use_trimap_refine"])

        # 人像 Matting 精修
        if "use_portrait_matting" in extra:
            kwargs["use_portrait_matting"] = bool(extra["use_portrait_matting"])
        if "portrait_matting_strength" in extra:
            kwargs["portrait_matting_strength"] = float(
                extra["portrait_matting_strength"]  # type: ignore[arg-type]
            )
        if "portrait_matting_model" in extra:
            kwargs["portrait_matting_model"] = str(extra["portrait_matting_model"])

        # Alpha 設定
        alpha_mode = str(extra.get("alpha_mode", "straight"))
        edge_decontam = bool(extra.get("edge_decontamination", True))
        kwargs["alpha_config"] = AlphaConfig(
            mode=AlphaMode(alpha_mode),
            edge_decontamination=edge_decontam,
        )

        # 解析度設定
        resolution = str(extra.get("resolution_mode", "1024"))
        kwargs["resolution_config"] = ResolutionConfig(
            mode=ResolutionMode(resolution),
        )

        return kwargs

    def _display_result(self, result: ProcessResult) -> None:
        """
        顯示處理結果

        Args:
            result: 處理結果
        """
        print("\n" + "=" * 60)
        print("✅ 處理完成！".center(60))
        print("=" * 60)
        print(f"\n  📊 總計: {result.total} 張圖片")
        print(f"  ✅ 成功: {result.success} 張")
        if result.failed > 0:
            print(f"  ❌ 失敗: {result.failed} 張")
        print(f"  📂 輸出: {result.output_folder}")
        print("\n" + "=" * 60 + "\n")
