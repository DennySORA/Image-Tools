"""
ImageProcessor 單元測試（使用 MagicMock）
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.core.interfaces import BackendProtocol
from src.core.processor import ImageProcessor
from src.data_model import ProcessConfig


def _make_config(
    input_folder: Path,
    output_folder: Path | None = None,
) -> ProcessConfig:
    """建立 ProcessConfig，繞過 frozen 限制"""
    return ProcessConfig(
        input_folder=input_folder,
        backend_name="mock",
        model="mock-v1",
        strength=0.5,
        output_folder=output_folder,
    )


class TestImageProcessorInit:
    """初始化測試"""

    @pytest.mark.unit
    def test_stores_backend(self) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        assert processor._backend is backend

    @pytest.mark.unit
    def test_default_max_workers(self) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        assert processor._max_workers == 1

    @pytest.mark.unit
    def test_custom_max_workers(self) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend, max_workers=4)
        assert processor._max_workers == 4

    @pytest.mark.unit
    def test_clamps_zero_workers(self) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend, max_workers=0)
        assert processor._max_workers == 1

    @pytest.mark.unit
    def test_clamps_negative_workers(self) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend, max_workers=-5)
        assert processor._max_workers == 1


class TestScanImages:
    """scan_images 測試"""

    @pytest.mark.unit
    def test_empty_folder(self, tmp_path: Path) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        assert processor.scan_images(tmp_path) == []

    @pytest.mark.unit
    def test_finds_png_files(self, valid_test_images_dir: Path) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        images = processor.scan_images(valid_test_images_dir)
        assert len(images) == 5
        assert all(p.suffix == ".png" for p in images)

    @pytest.mark.unit
    def test_finds_all_supported_extensions(self, tmp_path: Path) -> None:
        from PIL import Image

        extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]
        for ext in extensions:
            img = Image.new("RGB", (8, 8), color=(128, 128, 128))
            img.save(tmp_path / f"test{ext}")

        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        images = processor.scan_images(tmp_path)
        assert len(images) == len(extensions)

    @pytest.mark.unit
    def test_filters_non_images(self, mixed_test_images_dir: Path) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        images = processor.scan_images(mixed_test_images_dir)
        names = [p.name for p in images]
        assert "readme.txt" not in names

    @pytest.mark.unit
    def test_filters_directories(self, tmp_path: Path) -> None:
        from PIL import Image

        (tmp_path / "subdir").mkdir()
        img = Image.new("RGB", (8, 8), color=(128, 128, 128))
        img.save(tmp_path / "test.png")

        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        images = processor.scan_images(tmp_path)
        assert len(images) == 1
        assert images[0].name == "test.png"

    @pytest.mark.unit
    def test_returns_sorted(self, valid_test_images_dir: Path) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        images = processor.scan_images(valid_test_images_dir)
        names = [p.name for p in images]
        assert names == sorted(names)


class TestProcessFolder:
    """process_folder 測試"""

    @pytest.mark.unit
    def test_empty_folder_zero_result(self, tmp_path: Path) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        config = _make_config(input_dir, output_dir)
        result = processor.process_folder(config)
        assert result.total == 0
        assert result.success == 0
        assert result.failed == 0

    @pytest.mark.unit
    def test_empty_folder_skips_model_load(self, tmp_path: Path) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        config = _make_config(input_dir, output_dir)
        processor.process_folder(config)
        backend.load_model.assert_not_called()

    @pytest.mark.unit
    def test_single_image_success(self, tmp_path: Path) -> None:
        from PIL import Image

        input_dir = tmp_path / "images"
        input_dir.mkdir()
        Image.new("RGB", (8, 8)).save(input_dir / "test.png")
        output_dir = tmp_path / "output"

        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend)
        config = _make_config(input_dir, output_dir)
        result = processor.process_folder(config)
        assert result.total == 1
        assert result.success == 1

    @pytest.mark.unit
    def test_multiple_all_success(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.total == 5
        assert result.success == 5
        assert result.failed == 0

    @pytest.mark.unit
    def test_mixed_success_failure(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MagicMock(spec=BackendProtocol)
        backend.process.side_effect = [True, False, True, False, True]
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.total == 5
        assert result.success == 3
        assert result.failed == 2

    @pytest.mark.unit
    def test_creates_output_dir(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "new_output"
        assert not output_dir.exists()
        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        processor.process_folder(config)
        assert output_dir.exists()

    @pytest.mark.unit
    def test_calls_load_model_once(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        processor.process_folder(config)
        backend.load_model.assert_called_once()

    @pytest.mark.unit
    def test_output_folder_none_raises(self, tmp_path: Path) -> None:
        """output_folder=None 時 model_post_init 會自動設定，需要 mock 繞過"""
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)
        config = MagicMock()
        config.output_folder = None
        config.input_folder = tmp_path
        with pytest.raises(ValueError, match="Output folder is not set"):
            processor.process_folder(config)

    @pytest.mark.unit
    def test_result_output_folder_matches(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.output_folder == output_dir


class TestProcessingModeSelection:
    """處理模式選擇測試"""

    @pytest.mark.unit
    def test_sequential_single_worker(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend, max_workers=1)
        config = _make_config(valid_test_images_dir, output_dir)
        with patch.object(processor, "_process_sequential", return_value=5) as mock_seq:
            processor.process_folder(config)
            mock_seq.assert_called_once()

    @pytest.mark.unit
    def test_sequential_single_image(self, tmp_path: Path) -> None:
        from PIL import Image

        input_dir = tmp_path / "images"
        input_dir.mkdir()
        Image.new("RGB", (8, 8)).save(input_dir / "only.png")
        output_dir = tmp_path / "output"

        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend, max_workers=4)
        config = _make_config(input_dir, output_dir)
        with patch.object(processor, "_process_sequential", return_value=1) as mock_seq:
            processor.process_folder(config)
            mock_seq.assert_called_once()

    @pytest.mark.unit
    def test_parallel_multiple_workers(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend, max_workers=4)
        config = _make_config(valid_test_images_dir, output_dir)
        with patch.object(processor, "_process_parallel", return_value=5) as mock_par:
            processor.process_folder(config)
            mock_par.assert_called_once()


class TestProcessSingle:
    """process_single 測試"""

    @pytest.mark.unit
    def test_calls_ensure_model_loaded(self, tmp_path: Path) -> None:
        backend = MagicMock(spec=BackendProtocol)
        backend.process.return_value = True
        processor = ImageProcessor(backend)
        processor.process_single(tmp_path / "in.png", tmp_path / "out.png")
        backend.ensure_model_loaded.assert_called_once()

    @pytest.mark.unit
    def test_returns_backend_result(self, tmp_path: Path) -> None:
        backend = MagicMock(spec=BackendProtocol)
        processor = ImageProcessor(backend)

        backend.process.return_value = True
        assert processor.process_single(tmp_path / "a.png", tmp_path / "b.png") is True

        backend.process.return_value = False
        assert processor.process_single(tmp_path / "a.png", tmp_path / "b.png") is False
