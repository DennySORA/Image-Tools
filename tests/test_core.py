"""
核心模組測試

測試 core 模組中的類和函數
"""

from pathlib import Path

from src.data_model import (
    SUPPORTED_EXTENSIONS,
    ImageFile,
    ProcessConfig,
    ProcessResult,
    is_supported_image,
)


class TestProcessConfig:
    """測試 ProcessConfig"""

    def test_init_with_defaults(self, tmp_path: Path) -> None:
        """測試預設初始化"""
        config = ProcessConfig(
            input_folder=tmp_path,
            backend_name="ultra",
            model="auto",
            strength=0.7,
        )

        assert config.input_folder == tmp_path
        assert config.backend_name == "ultra"
        assert config.model == "auto"
        assert config.strength == 0.7
        assert config.output_folder == tmp_path / "output"
        assert config.extra_config == {}

    def test_custom_output_folder(self, tmp_path: Path) -> None:
        """測試自訂輸出資料夾"""
        custom_output = tmp_path / "custom_output"
        config = ProcessConfig(
            input_folder=tmp_path,
            backend_name="ultra",
            model="auto",
            strength=0.7,
            output_folder=custom_output,
        )

        assert config.output_folder == custom_output

    def test_extra_config(self, tmp_path: Path) -> None:
        """測試額外配置"""
        extra = {"color_filter": "green"}
        config = ProcessConfig(
            input_folder=tmp_path,
            backend_name="ultra",
            model="auto",
            strength=0.7,
            extra_config=extra,
        )

        assert config.extra_config == extra


class TestProcessResult:
    """測試 ProcessResult"""

    def test_success_rate(self, tmp_path: Path) -> None:
        """測試成功率計算"""
        result = ProcessResult(total=10, success=8, failed=2, output_folder=tmp_path)

        assert result.success_rate == 0.8

    def test_success_rate_zero_total(self, tmp_path: Path) -> None:
        """測試總數為 0 的情況"""
        result = ProcessResult(total=0, success=0, failed=0, output_folder=tmp_path)

        assert result.success_rate == 0.0

    def test_is_complete_success(self, tmp_path: Path) -> None:
        """測試是否全部成功"""
        success_result = ProcessResult(
            total=5, success=5, failed=0, output_folder=tmp_path
        )
        assert success_result.is_complete_success is True

        partial_result = ProcessResult(
            total=5, success=4, failed=1, output_folder=tmp_path
        )
        assert partial_result.is_complete_success is False


class TestImageFile:
    """測試 ImageFile"""

    def test_properties(self, tmp_path: Path) -> None:
        """測試屬性"""
        path = tmp_path / "test_image.jpg"
        path.touch()

        img_file = ImageFile(path=path)

        assert img_file.name == "test_image.jpg"
        assert img_file.stem == "test_image"
        assert img_file.suffix == ".jpg"


class TestImageSupport:
    """測試圖片格式支援"""

    def test_supported_extensions(self) -> None:
        """測試支援的副檔名列表"""
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS
        assert ".bmp" in SUPPORTED_EXTENSIONS
        assert ".gif" in SUPPORTED_EXTENSIONS

    def test_is_supported_image_valid(self, tmp_path: Path) -> None:
        """測試有效的圖片格式"""
        valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]

        for ext in valid_extensions:
            path = tmp_path / f"test{ext}"
            path.touch()
            assert is_supported_image(path) is True

    def test_is_supported_image_invalid(self, tmp_path: Path) -> None:
        """測試無效的圖片格式"""
        invalid_extensions = [".txt", ".pdf", ".doc", ".mp4"]

        for ext in invalid_extensions:
            path = tmp_path / f"test{ext}"
            path.touch()
            assert is_supported_image(path) is False

    def test_is_supported_image_nonexistent(self, tmp_path: Path) -> None:
        """測試不存在的文件"""
        path = tmp_path / "nonexistent.jpg"
        assert is_supported_image(path) is False

    def test_is_supported_image_directory(self, tmp_path: Path) -> None:
        """測試目錄"""
        dir_path = tmp_path / "directory.jpg"
        dir_path.mkdir()
        assert is_supported_image(dir_path) is False
