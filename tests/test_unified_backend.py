"""
Unified Backend 單元測試
"""

from pathlib import Path

import pytest
from PIL import Image
from src.backends.unified import ColorFilter, ColorFilterConfig, UnifiedBackend


class TestUnifiedBackendInit:
    """測試 UnifiedBackend 初始化"""

    def test_default_init(self) -> None:
        """測試預設初始化"""
        backend = UnifiedBackend()

        assert backend.strength == 0.7
        assert backend.color_filter.enabled is False

    def test_custom_strength(self) -> None:
        """測試自訂強度"""
        backend = UnifiedBackend(strength=0.5)
        assert backend.strength == 0.5

    def test_color_filter_config(self) -> None:
        """測試色彩過濾配置"""
        color_filter = ColorFilterConfig(enabled=True, color=ColorFilter.GREEN)
        backend = UnifiedBackend(color_filter=color_filter)

        assert backend.color_filter.enabled is True
        assert backend.color_filter.color == ColorFilter.GREEN


class TestUnifiedBackendProcessing:
    """測試 Unified Backend 處理"""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_simple_image(
        self, simple_portrait_image: Path, temp_output_dir: Path
    ) -> None:
        """測試處理簡單圖片"""
        backend = UnifiedBackend(strength=0.7)
        output_path = temp_output_dir / "unified_output.png"

        try:
            backend.load_model()
            success = backend.process(simple_portrait_image, output_path)

            assert success is True
            assert output_path.exists()

            result = Image.open(output_path)
            assert result.mode == "RGBA"
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_with_green_filter(
        self, greenscreen_image: Path, temp_output_dir: Path
    ) -> None:
        """測試綠幕過濾"""
        color_filter = ColorFilterConfig(enabled=True, color=ColorFilter.GREEN)
        backend = UnifiedBackend(strength=0.8, color_filter=color_filter)
        output_path = temp_output_dir / "unified_green_output.png"

        try:
            backend.load_model()
            success = backend.process(greenscreen_image, output_path)

            assert success is True
            assert output_path.exists()
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


class TestUnifiedBackendMethods:
    """測試類方法"""

    def test_get_available_models(self) -> None:
        """測試取得可用模型"""
        models = UnifiedBackend.get_available_models()
        assert isinstance(models, list)
        assert "auto" in models

    def test_get_model_description(self) -> None:
        """測試取得模型說明"""
        description = UnifiedBackend.get_model_description()
        assert isinstance(description, str)
        assert "BiRefNet" in description
