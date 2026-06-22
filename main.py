#!/usr/bin/env python3
"""
圖片處理工具

主程式進入點，使用現代化 CLI 介面

使用方法:
    uv run main.py
"""

import logging
import sys
from importlib import import_module

from src.app import ApplicationService


def _register_backends() -> None:
    """確保後端模組在應用程式啟動前完成註冊。"""
    backend_modules = [
        "src.backends.ultra",
        "src.backends.gemini_watermark",
        "src.backends.image_splitter",
        "src.backends.batch_compare",
    ]
    for module_path in backend_modules:
        import_module(module_path)


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

    _register_backends()

    # 建立並執行應用程式服務
    app = ApplicationService()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
