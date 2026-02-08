#!/usr/bin/env python3
"""
圖片處理工具

主程式進入點，使用現代化 CLI 介面

使用方法:
    uv run main.py
"""

import logging
import sys

from src.backends import BackendRegistry
from src.core.processor import ImageProcessor
from src.ui import ModernUI


def main() -> int:
    """
    主程式

    Returns:
        退出碼 (0: 成功, 1: 失敗或取消)
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        ui = ModernUI()

        # 主循環 - 支援連續處理（自動返回主選單）
        while True:
            # 1. 執行互動式設定流程（使用新的現代化 UI）
            config = ui.run()

            if config is None:
                # 使用者取消，退出程式
                print("\n👋 再見！")
                return 0

            # 2. 顯示處理摘要
            ui.show_summary(config)

            # 3. 建立後端
            # 如果是 unified/ultra 後端且有 color_filter 設定，需要建立 ColorFilterConfig
            backend_kwargs = {}
            if config.backend_name in ("unified", "ultra") and "color_filter" in config.extra_config:
                from src.common import ColorFilter, ColorFilterConfig

                color_filter = ColorFilterConfig(
                    enabled=True,
                    color=ColorFilter(config.extra_config["color_filter"]),
                    edge_refine_strength=config.strength,  # 使用主強度參數
                )
                backend_kwargs["color_filter"] = color_filter

            backend = BackendRegistry.create(
                name=config.backend_name,
                model=config.model,
                strength=config.strength,
                **backend_kwargs,
            )

            # 4. 建立處理器並處理圖片
            processor = ImageProcessor(backend)
            result = processor.process_folder(config)

            # 5. 顯示結果
            print("\n" + "=" * 60)
            print("✅ 處理完成！".center(60))
            print("=" * 60)
            print(f"\n  📊 總計: {result.total} 張圖片")
            print(f"  ✅ 成功: {result.success} 張")
            if result.failed > 0:
                print(f"  ❌ 失敗: {result.failed} 張")
            print(f"  📂 輸出: {result.output_folder}")
            print("\n" + "=" * 60 + "\n")

            # 6. 自動返回主選單（不詢問）
            print("🔄 返回主選單...\n")

    except KeyboardInterrupt:
        print("\n\n👋 已中斷操作，再見！")
        return 130

    except Exception as exc:
        print(f"\n❌ 錯誤: {exc}\n")
        logging.exception("處理時發生錯誤")
        return 1


if __name__ == "__main__":
    sys.exit(main())
