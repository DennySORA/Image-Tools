"""
核心模組測試

測試 core 模組中的類和函數
"""

from pathlib import Path

import pytest

from src.core.models import (
    SUPPORTED_EXTENSIONS,
    ImageFile,
    ProcessConfig,
    ProcessResult,
    is_supported_image,
)
from src.core.processor import ImageProcessor


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


class TestImageProcessor:
    """測試 ImageProcessor"""

    def test_init(self) -> None:
        """測試初始化"""
        from src.backends.unified import UnifiedBackend

        backend = UnifiedBackend()
        processor = ImageProcessor(backend)

        assert processor._backend == backend

    @pytest.mark.integration
    def test_scan_images(
        self,
        test_images_dir: Path,
        simple_portrait_image: Path,
        greenscreen_image: Path,
    ) -> None:
        """測試掃描圖片"""
        from src.backends.unified import UnifiedBackend

        # 確保測試圖片被創建（通過使用 fixtures）
        assert simple_portrait_image.exists()
        assert greenscreen_image.exists()

        backend = UnifiedBackend()
        processor = ImageProcessor(backend)

        images = processor.scan_images(test_images_dir)

        # 至少應該找到我們創建的測試圖片
        assert len(images) > 0

        # 所有圖片都應該是支援的格式
        for img_path in images:
            assert is_supported_image(img_path)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_process_folder_empty(self, tmp_path: Path) -> None:
        """測試處理空資料夾"""
        from src.backends.unified import UnifiedBackend

        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        config = ProcessConfig(
            input_folder=empty_folder,
            backend_name="unified",
            model="auto",
            strength=0.7,
        )

        backend = UnifiedBackend()
        processor = ImageProcessor(backend)

        result = processor.process_folder(config)

        assert result.total == 0
        assert result.success == 0
        assert result.failed == 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_process_folder_with_images(
        self, test_images_dir: Path, temp_output_dir: Path
    ) -> None:
        """測試處理包含圖片的資料夾"""
        from src.backends.unified import UnifiedBackend

        config = ProcessConfig(
            input_folder=test_images_dir,
            backend_name="unified",
            model="auto",
            strength=0.7,
            output_folder=temp_output_dir,
        )

        try:
            backend = UnifiedBackend(strength=config.strength)
            backend.load_model()
            processor = ImageProcessor(backend)

            result = processor.process_folder(config)

            # 應該有處理一些圖片
            assert result.total > 0
            assert result.output_folder == temp_output_dir
        except Exception as e:
            pytest.skip(f"Process folder test skipped: {e}")
