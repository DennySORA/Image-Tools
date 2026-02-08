"""
端到端測試：預設配置系統

使用自動生成的測試圖像驗證所有預設配置的完整處理流程
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common import ColorFilter, ColorFilterConfig
from src.features.background_removal.ultra import UltraBackend
from tests.fixtures.synthetic.generate_test_images import (
    create_complex_edges_image,
    create_gradient_background_image,
    create_portrait_like_image,
    create_solid_background_image,
)


@pytest.fixture(scope="module")
def test_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """創建測試輸出目錄"""
    output_dir = tmp_path_factory.mktemp("preset_test_outputs")
    yield output_dir
    # 清理（可選）
    # shutil.rmtree(output_dir)


@pytest.fixture(scope="module")
def test_images_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """創建並生成測試圖像目錄"""
    images_dir = tmp_path_factory.mktemp("preset_test_images")

    # 生成測試圖像
    test_cases = {
        "green_bg": create_solid_background_image((512, 512), (0, 255, 0)),
        "white_bg": create_solid_background_image((512, 512), (255, 255, 255)),
        "black_bg": create_solid_background_image((512, 512), (0, 0, 0)),
        "portrait": create_portrait_like_image((512, 512)),
        "gradient": create_gradient_background_image((512, 512)),
        "complex": create_complex_edges_image((512, 512)),
    }

    for name, image in test_cases.items():
        Image.fromarray(image).save(images_dir / f"{name}.png")

    return images_dir


class TestPresetBasicProcessing:
    """測試預設配置的基本處理功能"""

    @pytest.mark.slow
    def test_balanced_preset_processes_image(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 Balanced 預設處理圖像"""
        backend = UltraBackend.from_preset("balanced")
        backend.load_model()

        input_path = test_images_dir / "green_bg.png"
        output_path = test_output_dir / "balanced_green_bg.png"

        success = backend.process(input_path, output_path)

        assert success is True
        assert output_path.exists()

        # 驗證輸出是 RGBA
        output_image = Image.open(output_path)
        assert output_image.mode == "RGBA"
        assert output_image.size == (512, 512)

    @pytest.mark.slow
    def test_high_preset_processes_portrait(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 High 預設處理人像"""
        backend = UltraBackend.from_preset("high")
        backend.load_model()

        input_path = test_images_dir / "portrait.png"
        output_path = test_output_dir / "high_portrait.png"

        success = backend.process(input_path, output_path)

        assert success is True
        assert output_path.exists()

        # 驗證輸出
        output_image = Image.open(output_path)
        assert output_image.mode == "RGBA"

    @pytest.mark.slow
    def test_ultra_preset_processes_complex(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 Ultra 預設處理複雜圖像"""
        backend = UltraBackend.from_preset("ultra")
        backend.load_model()

        input_path = test_images_dir / "complex.png"
        output_path = test_output_dir / "ultra_complex.png"

        success = backend.process(input_path, output_path)

        assert success is True
        assert output_path.exists()


class TestPresetWithColorFilter:
    """測試預設配置搭配色彩過濾"""

    @pytest.mark.slow
    def test_balanced_with_green_filter(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 Balanced + 綠幕過濾"""
        green_filter = ColorFilterConfig(color=ColorFilter.GREEN, enabled=True)
        backend = UltraBackend.from_preset("balanced", color_filter=green_filter)
        backend.load_model()

        input_path = test_images_dir / "green_bg.png"
        output_path = test_output_dir / "balanced_green_filtered.png"

        success = backend.process(input_path, output_path)

        assert success is True
        assert output_path.exists()

    @pytest.mark.slow
    def test_high_with_white_filter(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 High + 白背景過濾"""
        white_filter = ColorFilterConfig(color=ColorFilter.WHITE, enabled=True)
        backend = UltraBackend.from_preset("high", color_filter=white_filter)
        backend.load_model()

        input_path = test_images_dir / "white_bg.png"
        output_path = test_output_dir / "high_white_filtered.png"

        success = backend.process(input_path, output_path)

        assert success is True
        assert output_path.exists()

    @pytest.mark.slow
    def test_ultra_with_black_filter(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 Ultra + 黑背景過濾"""
        black_filter = ColorFilterConfig(color=ColorFilter.BLACK, enabled=True)
        backend = UltraBackend.from_preset("ultra", color_filter=black_filter)
        backend.load_model()

        input_path = test_images_dir / "black_bg.png"
        output_path = test_output_dir / "ultra_black_filtered.png"

        success = backend.process(input_path, output_path)

        assert success is True
        assert output_path.exists()


class TestPresetAlphaQuality:
    """測試預設配置的 Alpha 品質"""

    @pytest.mark.slow
    def test_balanced_alpha_range(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 Balanced 輸出的 alpha 範圍"""
        backend = UltraBackend.from_preset("balanced")
        backend.load_model()

        input_path = test_images_dir / "green_bg.png"
        output_path = test_output_dir / "balanced_alpha_test.png"

        backend.process(input_path, output_path)

        # 檢查 alpha 通道
        output_image = Image.open(output_path)
        alpha_channel = np.array(output_image)[:, :, 3]

        # Alpha 應該有 0-255 的完整範圍
        assert alpha_channel.min() >= 0
        assert alpha_channel.max() <= 255

        # 應該有前景（高 alpha）和背景（低 alpha）
        has_foreground = np.any(alpha_channel > 200)
        has_background = np.any(alpha_channel < 50)

        assert has_foreground, "應該有前景區域"
        assert has_background, "應該有背景區域"

    @pytest.mark.slow
    def test_high_vs_balanced_alpha_difference(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 High 和 Balanced 的 alpha 差異"""
        input_path = test_images_dir / "portrait.png"

        # Balanced 處理
        backend_balanced = UltraBackend.from_preset("balanced")
        backend_balanced.load_model()
        output_balanced = test_output_dir / "alpha_compare_balanced.png"
        backend_balanced.process(input_path, output_balanced)

        # High 處理
        backend_high = UltraBackend.from_preset("high")
        backend_high.load_model()
        output_high = test_output_dir / "alpha_compare_high.png"
        backend_high.process(input_path, output_high)

        # 比較結果
        alpha_balanced = np.array(Image.open(output_balanced))[:, :, 3]
        alpha_high = np.array(Image.open(output_high))[:, :, 3]

        # 兩者應該不完全相同（因為 High 有人像精修）
        difference = np.abs(alpha_balanced.astype(float) - alpha_high.astype(float))
        mean_diff = difference.mean()

        # 應該有一定差異（但不會太大）
        assert mean_diff > 0, "High 和 Balanced 應該有差異"
        assert mean_diff < 50, "差異不應該太大"


class TestPresetResolutionHandling:
    """測試預設配置的解析度處理"""

    @pytest.mark.slow
    def test_balanced_uses_1024_resolution(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 Balanced 使用 1024 解析度"""
        backend = UltraBackend.from_preset("balanced")
        assert backend.resolution_config.mode.value == "1024"

    @pytest.mark.slow
    def test_high_uses_1536_resolution(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 High 使用 1536 解析度"""
        backend = UltraBackend.from_preset("high")
        assert backend.resolution_config.mode.value == "1536"

    @pytest.mark.slow
    def test_ultra_uses_2048_resolution(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試 Ultra 使用 2048 解析度"""
        backend = UltraBackend.from_preset("ultra")
        assert backend.resolution_config.mode.value == "2048"


class TestPresetRobustness:
    """測試預設配置的魯棒性"""

    @pytest.mark.slow
    def test_all_presets_handle_different_backgrounds(
        self, test_images_dir: Path, test_output_dir: Path
    ) -> None:
        """測試所有預設處理不同背景"""
        test_cases = ["green_bg", "white_bg", "black_bg", "gradient"]

        for preset_name in ["balanced", "high", "ultra"]:
            backend = UltraBackend.from_preset(preset_name)
            backend.load_model()

            for case in test_cases:
                input_path = test_images_dir / f"{case}.png"
                output_path = test_output_dir / f"{preset_name}_{case}_robust.png"

                success = backend.process(input_path, output_path)

                assert success is True, f"{preset_name} 處理 {case} 失敗"
                assert output_path.exists(), f"{preset_name} 未生成 {case} 輸出"

    @pytest.mark.slow
    def test_presets_handle_missing_input(
        self, test_output_dir: Path
    ) -> None:
        """測試預設處理不存在的輸入"""
        backend = UltraBackend.from_preset("balanced")
        backend.load_model()

        input_path = Path("nonexistent.png")
        output_path = test_output_dir / "should_not_exist.png"

        success = backend.process(input_path, output_path)

        assert success is False
        assert not output_path.exists()


class TestPresetPerformanceCharacteristics:
    """測試預設配置的性能特徵"""

    @pytest.mark.slow
    def test_preset_model_loading_idempotent(self) -> None:
        """測試預設的模型載入是冪等的"""
        backend = UltraBackend.from_preset("balanced")

        # 第一次載入
        backend.load_model()
        assert backend._model is not None

        # 第二次載入應該不報錯
        backend.load_model()
        assert backend._model is not None

    @pytest.mark.slow
    def test_all_presets_initialize_successfully(self) -> None:
        """測試所有預設都能成功初始化"""
        for preset_name in ["balanced", "high", "ultra"]:
            try:
                backend = UltraBackend.from_preset(preset_name)
                backend.load_model()
                success = True
            except Exception:
                success = False

            assert success, f"{preset_name} 預設初始化失敗"


@pytest.mark.parametrize("preset_name", ["balanced", "high", "ultra"])
@pytest.mark.slow
def test_preset_produces_valid_rgba_output(
    preset_name: str, test_images_dir: Path, test_output_dir: Path
) -> None:
    """參數化測試：所有預設都產生有效的 RGBA 輸出"""
    backend = UltraBackend.from_preset(preset_name)
    backend.load_model()

    input_path = test_images_dir / "green_bg.png"
    output_path = test_output_dir / f"{preset_name}_rgba_test.png"

    success = backend.process(input_path, output_path)

    assert success is True
    assert output_path.exists()

    output_image = Image.open(output_path)
    assert output_image.mode == "RGBA"
    assert output_image.size == (512, 512)

    # 檢查有 4 個通道
    output_array = np.array(output_image)
    assert output_array.shape[2] == 4
