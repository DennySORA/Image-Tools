"""
控制台工具模組

提供控制台互動的基礎工具函數，遵循單一職責原則
"""

import sys
from typing import TextIO


class Console:
    """
    控制台工具類別

    封裝所有控制台相關的操作
    """

    @staticmethod
    def _write(
        message: str,
        *,
        end: str = "\n",
        file: TextIO = sys.stdout,
        flush: bool = False,
    ) -> None:
        file.write(f"{message}{end}")
        if flush:
            file.flush()

    @staticmethod
    def clear() -> None:
        """清除螢幕"""
        Console._write("\033[2J\033[H", end="", flush=True)

    @staticmethod
    def write_line(message: str) -> None:
        """輸出單行文字"""
        Console._write(message)

    @staticmethod
    def write_error(message: str) -> None:
        """輸出錯誤文字到 stderr"""
        Console._write(message, file=sys.stderr)

    @staticmethod
    def print_header(title: str, width: int = 60) -> None:
        """
        顯示標題

        Args:
            title: 標題文字
            width: 寬度
        """
        Console._write("=" * width)
        Console._write(title.center(width))
        Console._write("=" * width)
        Console._write("")

    @staticmethod
    def print_section(title: str, width: int = 40) -> None:
        """
        顯示區段標題

        Args:
            title: 標題文字
            width: 寬度
        """
        Console._write(f"\n{title}")
        Console._write("-" * width)

    @staticmethod
    def print_separator(width: int = 50) -> None:
        """顯示分隔線"""
        Console._write("-" * width)

    @staticmethod
    def get_input(prompt: str, default: str | None = None) -> str:
        """
        取得使用者輸入

        Args:
            prompt: 提示文字
            default: 預設值

        Returns:
            使用者輸入或預設值
        """
        if default:
            result = input(f"{prompt} [{default}]: ").strip()
            return result if result else default
        return input(f"{prompt}: ").strip()

    @staticmethod
    def get_choice(prompt: str, options: list[str], default: int = 1) -> int:
        """
        取得使用者選擇

        Args:
            prompt: 提示文字
            options: 選項列表
            default: 預設選項 (1-based)

        Returns:
            使用者選擇的索引 (1-based)
        """
        Console._write(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            marker = " *" if i == default else ""
            Console._write(f"  {i}. {option}{marker}")

        while True:
            choice = input(f"\n請選擇 [1-{len(options)}] (預設: {default}): ").strip()
            if not choice:
                return default
            try:
                idx = int(choice)
                if 1 <= idx <= len(options):
                    return idx
            except ValueError:
                pass
            Console._write(f"請輸入 1 到 {len(options)} 之間的數字")

    @staticmethod
    def get_number(
        prompt: str,
        min_val: float,
        max_val: float,
        default: float,
    ) -> float:
        """
        取得數值輸入

        Args:
            prompt: 提示文字
            min_val: 最小值
            max_val: 最大值
            default: 預設值

        Returns:
            使用者輸入的數值
        """
        while True:
            value = input(f"{prompt} [{min_val}-{max_val}] (預設: {default}): ").strip()
            if not value:
                return default
            try:
                num = float(value)
                if min_val <= num <= max_val:
                    return num
                Console._write(f"請輸入 {min_val} 到 {max_val} 之間的數值")
            except ValueError:
                Console._write("請輸入有效的數字")

    @staticmethod
    def confirm(prompt: str, default: bool = True) -> bool:
        """
        確認對話

        Args:
            prompt: 提示文字
            default: 預設值

        Returns:
            使用者確認結果
        """
        default_hint = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_hint}]: ").strip().lower()

        if not response:
            return default
        return response in ("y", "yes", "是")

    @staticmethod
    def wait_for_key(prompt: str = "按 Enter 繼續...") -> None:
        """等待使用者按鍵"""
        input(f"\n{prompt}")
