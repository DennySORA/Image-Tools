"""
ImageProcessor 端對端測試（使用 MockBackend + 真實圖片 I/O）
"""

from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image
from src.core.interfaces import BaseBackend
from src.core.processor import ImageProcessor
from src.data_model import ProcessConfig


class MockBackend(BaseBackend):
    """E2E 測試用 MockBackend（讀取圖片 → RGBA → 儲存 PNG）"""

    name: ClassVar[str] = "mock"
    description: ClassVar[str] = "Mock backend for E2E testing"

    def __init__(
        self,
        strength: float = 0.5,
        fail_files: set[str] | None = None,
    ) -> None:
        super().__init__(strength=strength)
        self.fail_files: set[str] = fail_files or set()
        self.load_model_calls: int = 0
        self.process_calls: int = 0
        self.processed_files: list[str] = []

    def load_model(self) -> None:
        self._model_loaded = True
        self.load_model_calls += 1

    def process(self, input_path: Path, output_path: Path) -> bool:
        self.process_calls += 1
        self.processed_files.append(input_path.name)
        if input_path.name in self.fail_files:
            return False
        try:
            img = Image.open(input_path)
            img_rgba = img.convert("RGBA")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_rgba.save(output_path, "PNG")
            return True
        except Exception:
            return False

    @classmethod
    def get_available_models(cls) -> list[str]:
        return ["mock-v1"]

    @classmethod
    def get_model_description(cls) -> str:
        return "Mock model for E2E testing"


def _make_config(
    input_folder: Path,
    output_folder: Path,
) -> ProcessConfig:
    return ProcessConfig(
        input_folder=input_folder,
        backend_name="mock",
        model="mock-v1",
        strength=0.5,
        output_folder=output_folder,
    )


class TestFullPipelineE2E:
    """完整流程端對端測試"""

    @pytest.mark.e2e
    def test_batch_all_success(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.success == 5

    @pytest.mark.e2e
    def test_output_files_exist(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        processor.process_folder(config)
        for img_path in valid_test_images_dir.glob("*.png"):
            assert (output_dir / f"{img_path.stem}.png").exists()

    @pytest.mark.e2e
    def test_output_files_are_rgba(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        processor.process_folder(config)
        for out_file in output_dir.glob("*.png"):
            img = Image.open(out_file)
            assert img.mode == "RGBA"

    @pytest.mark.e2e
    def test_non_image_filtered(
        self, mixed_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend)
        config = _make_config(mixed_test_images_dir, output_dir)
        processor.process_folder(config)
        # 非圖片檔不應被處理
        assert "readme.txt" not in backend.processed_files

    @pytest.mark.e2e
    def test_corrupted_image_fails_gracefully(
        self, mixed_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend)
        config = _make_config(mixed_test_images_dir, output_dir)
        result = processor.process_folder(config)
        # corrupted.png 有 .png 副檔名所以會被掃描
        assert "corrupted.png" in backend.processed_files
        # 但它無法被 Image.open() 開啟，MockBackend 會回傳 False
        assert result.failed >= 1

    @pytest.mark.e2e
    def test_result_success_rate(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.success_rate == 1.0

    @pytest.mark.e2e
    def test_result_is_complete_success(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.is_complete_success is True


class TestParallelProcessingE2E:
    """並行處理端對端測試"""

    @pytest.mark.e2e
    def test_parallel_completes_all(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend, max_workers=3)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.success == 5
        assert result.total == 5

    @pytest.mark.e2e
    def test_parallel_matches_sequential(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_seq = tmp_path / "output_seq"
        backend_seq = MockBackend()
        proc_seq = ImageProcessor(backend_seq, max_workers=1)
        config_seq = _make_config(valid_test_images_dir, output_seq)
        result_seq = proc_seq.process_folder(config_seq)

        output_par = tmp_path / "output_par"
        backend_par = MockBackend()
        proc_par = ImageProcessor(backend_par, max_workers=3)
        config_par = _make_config(valid_test_images_dir, output_par)
        result_par = proc_par.process_folder(config_par)

        assert result_seq.total == result_par.total
        assert result_seq.success == result_par.success
        assert result_seq.failed == result_par.failed

    @pytest.mark.e2e
    def test_parallel_thread_safety(self, tmp_path: Path) -> None:
        """10+ 張圖片並行處理，驗證 success + failed == total"""
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        for i in range(12):
            Image.new("RGB", (32, 32), color=(i * 20, 100, 200)).save(
                input_dir / f"img_{i:02d}.png"
            )
        output_dir = tmp_path / "output"

        backend = MockBackend()
        processor = ImageProcessor(backend, max_workers=4)
        config = _make_config(input_dir, output_dir)
        result = processor.process_folder(config)
        assert result.success + result.failed == result.total
        assert result.total == 12


class TestMixedResultsE2E:
    """混合結果端對端測試"""

    @pytest.mark.e2e
    def test_partial_failure(self, valid_test_images_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        images = sorted(valid_test_images_dir.glob("*.png"))
        fail_names = {images[0].name, images[1].name}
        backend = MockBackend(fail_files=fail_names)
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.failed == 2
        assert result.success == 3

    @pytest.mark.e2e
    def test_processing_continues_after_failure(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        images = sorted(valid_test_images_dir.glob("*.png"))
        # 第一張失敗，後續仍應處理
        backend = MockBackend(fail_files={images[0].name})
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        result = processor.process_folder(config)
        assert result.total == 5
        assert backend.process_calls == 5

    @pytest.mark.e2e
    def test_failed_files_no_output(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        images = sorted(valid_test_images_dir.glob("*.png"))
        fail_name = images[0].name
        backend = MockBackend(fail_files={fail_name})
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        processor.process_folder(config)
        # fail_files 設定會讓 MockBackend.process 回傳 False 但不寫檔
        assert not (output_dir / f"{images[0].stem}.png").exists()

    @pytest.mark.e2e
    def test_successful_files_have_output(
        self, valid_test_images_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        images = sorted(valid_test_images_dir.glob("*.png"))
        fail_name = images[0].name
        backend = MockBackend(fail_files={fail_name})
        processor = ImageProcessor(backend)
        config = _make_config(valid_test_images_dir, output_dir)
        processor.process_folder(config)
        for img_path in images[1:]:
            assert (output_dir / f"{img_path.stem}.png").exists()


class TestEmptyAndEdgeCasesE2E:
    """空資料夾和邊界情況端對端測試"""

    @pytest.mark.e2e
    def test_empty_folder(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        backend = MockBackend()
        processor = ImageProcessor(backend)
        config = _make_config(input_dir, output_dir)
        result = processor.process_folder(config)
        assert result.total == 0
        assert backend.load_model_calls == 0
