"""
歷史記錄模組

管理使用者曾經使用過的資料夾路徑和後端設定，提供快速選擇功能
"""

import json
from pathlib import Path
from typing import Any


_HISTORY_FILE = ".rembg_history.json"
_SETTINGS_FILE = ".ultra_settings.json"
_MAX_ENTRIES = 10


class PathHistory:
    """
    路徑歷史管理

    負責讀寫路徑歷史記錄到 JSON 檔案
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """
        初始化路徑歷史

        Args:
            base_dir: 歷史檔案所在目錄，預設為目前工作目錄
        """
        root = base_dir or Path.cwd()
        self._history_file = root / _HISTORY_FILE

    def load(self) -> list[Path]:
        """
        讀取歷史路徑列表

        自動過濾已不存在的路徑

        Returns:
            有效的歷史路徑列表（最新在前）
        """
        if not self._history_file.exists():
            return []

        try:
            data = json.loads(self._history_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(data, list):
            return []

        paths = [Path(p) for p in data if isinstance(p, str)]
        return [p for p in paths if p.is_dir()]

    def save(self, path: Path) -> None:
        """
        新增路徑到歷史

        去重並將最新路徑排在最前面，最多保留 10 條

        Args:
            path: 要儲存的路徑
        """
        resolved = path.resolve()
        existing = self.load()

        # 去重：移除已存在的相同路徑
        entries = [p for p in existing if p.resolve() != resolved]
        # 最新排前面
        entries.insert(0, resolved)
        # 限制數量
        entries = entries[:_MAX_ENTRIES]

        data = [str(p) for p in entries]
        self._history_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class SettingsHistory:
    """
    後端設定歷史管理

    負責讀寫後端設定到 JSON 檔案，用於記住上一次的設定
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """
        初始化設定歷史

        Args:
            base_dir: 設定檔案所在目錄，預設為目前工作目錄
        """
        root = base_dir or Path.cwd()
        self._settings_file = root / _SETTINGS_FILE

    def load(self) -> dict[str, Any] | None:
        """
        讀取上一次的設定

        Returns:
            設定字典，若無歷史則返回 None
        """
        if not self._settings_file.exists():
            return None

        try:
            data = json.loads(self._settings_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        if not isinstance(data, dict):
            return None

        return data

    def save(self, settings: dict[str, Any]) -> None:
        """
        儲存設定

        Args:
            settings: 設定字典
        """
        self._settings_file.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
