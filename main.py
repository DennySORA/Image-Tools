#!/usr/bin/env python3
"""
圖片處理工具

主程式進入點，使用現代化 CLI 介面

使用方法:
    uv run main.py
"""

import logging
import sys

from src.app import ApplicationService


def main() -> int:
    """
    主程式進入點

    使用依賴注入模式，將業務邏輯委託給 ApplicationService

    Returns:
        退出碼 (0: 成功, 1: 失敗, 130: 中斷)
    """
    # 配置日誌
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # 建立並執行應用程式服務
    app = ApplicationService()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
