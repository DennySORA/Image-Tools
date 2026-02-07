#!/usr/bin/env python3
"""
圖片背景移除工具

主程式進入點，負責組合各模組並啟動應用程式
遵循依賴反轉原則 (DIP)：依賴抽象而非具體實作

使用方法:
    uv run main.py
"""

import logging
import sys

from src.backends import BackendRegistry
from src.core.processor import ImageProcessor
from src.ui import InteractiveUI


def main() -> int:
    """
    主程式

    Returns:
        退出碼 (0: 成功, 1: 失敗或取消)
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        # 建立 UI
        ui = InteractiveUI()
        last_result_success = True

        # 主循環 - 支援連續處理
        while True:
            # 1. 執行交互式設定流程
            config = ui.run()

            if config is None:
                ui.show_cancelled()
                break

            # 2. 建立後端
            backend = BackendRegistry.create(
                name=config.backend_name,
                model=config.model,
                strength=config.strength,
            )

            # 3. 建立處理器並處理圖片
            processor = ImageProcessor(backend)
            result = processor.process_folder(config)
            last_result_success = result.is_complete_success

            # 4. 顯示結果
            ui.show_result(result)

            # 5. 詢問是否繼續
            if not ui.ask_continue():
                break

    except KeyboardInterrupt:
        sys.stdout.write("\n\n已取消\n")
        sys.stdout.flush()
        return 1

    except Exception as exc:
        sys.stderr.write(f"\n錯誤: {exc}\n")
        sys.stderr.flush()
        return 1
    else:
        return 0 if last_result_success else 1


if __name__ == "__main__":
    sys.exit(main())
