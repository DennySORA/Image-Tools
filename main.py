#!/usr/bin/env python3
"""
圖片背景移除工具

主程式進入點，負責組合各模組並啟動應用程式
遵循依賴反轉原則 (DIP)：依賴抽象而非具體實作

使用方法:
    uv run main.py
"""

import sys

from src.core.processor import ImageProcessor
from src.backends import BackendRegistry
from src.ui import InteractiveUI


def main() -> int:
    """
    主程式

    Returns:
        退出碼 (0: 成功, 1: 失敗或取消)
    """
    try:
        # 1. 建立 UI
        ui = InteractiveUI()

        # 2. 執行交互式設定流程
        config = ui.run()

        if config is None:
            ui.show_cancelled()
            return 1

        # 3. 建立後端
        backend = BackendRegistry.create(
            name=config.backend_name,
            model=config.model,
            strength=config.strength,
        )

        # 4. 建立處理器並處理圖片
        processor = ImageProcessor(backend)
        result = processor.process_folder(config)

        # 5. 顯示結果
        ui.show_result(result)
        ui.wait_for_exit()

        return 0 if result.is_complete_success else 1

    except KeyboardInterrupt:
        print("\n\n已取消")
        return 1

    except Exception as e:
        print(f"\n錯誤: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
