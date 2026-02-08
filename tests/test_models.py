"""
模型類別簡單測試
"""

from pathlib import Path

import pytest

from src.core.models import ProcessConfig, is_supported_image


def test_process_config_default_output(tmp_path: Path) -> None:
    """測試 ProcessConfig 預設輸出路徑"""
    config = ProcessConfig(
        input_folder=tmp_path,
        backend_name="ultra",
        model="auto",
        strength=0.7,
    )
    assert config.output_folder == tmp_path / "output"


def test_is_supported_image(tmp_path: Path) -> None:
    """測試支援的圖片格式"""
    jpg_file = tmp_path / "test.jpg"
    jpg_file.touch()
    assert is_supported_image(jpg_file) is True

    txt_file = tmp_path / "test.txt"
    txt_file.touch()
    assert is_supported_image(txt_file) is False
