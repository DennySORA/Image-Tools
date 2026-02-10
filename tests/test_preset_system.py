"""
測試預設配置系統

驗證所有預設配置的正確性和功能完整性
"""

import pytest

from src.common import (
    ColorFilter,
    ColorFilterConfig,
    PresetLevel,
    get_preset,
    list_presets,
)
from src.common.alpha_config import AlphaMode, ResolutionMode
from src.features.background_removal.ultra import UltraBackend


class TestPresetDefinitions:
    """測試預設配置的定義"""

    def test_balanced_preset_values(self) -> None:
        """測試 Balanced 預設的數值"""
        preset = get_preset(PresetLevel.BALANCED)

        assert preset.strength == 0.6
        assert preset.resolution_mode == ResolutionMode.FIXED_1024
        assert preset.use_trimap_refine is True
        assert preset.edge_decontamination is True
        assert preset.decontamination_strength == 0.6

    def test_high_preset_values(self) -> None:
        """測試 High 預設的數值"""
        preset = get_preset(PresetLevel.HIGH)

        assert preset.strength == 0.75
        assert preset.resolution_mode == ResolutionMode.FIXED_1536
        assert preset.use_trimap_refine is True
        assert preset.edge_decontamination is True
        assert preset.decontamination_strength == 0.75

    def test_ultra_preset_values(self) -> None:
        """測試 Ultra 預設的數值"""
        preset = get_preset(PresetLevel.ULTRA)

        assert preset.strength == 0.95
        assert preset.resolution_mode == ResolutionMode.FIXED_2048
        assert preset.use_trimap_refine is True
        assert preset.edge_decontamination is True
        assert preset.decontamination_strength == 0.95

    def test_preset_level_names(self) -> None:
        """測試預設等級名稱"""
        balanced = get_preset("balanced")
        assert balanced.level_name == "Balanced"

        high = get_preset("high")
        assert high.level_name == "High"

        ultra = get_preset("ultra")
        assert ultra.level_name == "Ultra"

    def test_preset_descriptions(self) -> None:
        """測試預設描述"""
        balanced = get_preset("balanced")
        assert "日常使用" in balanced.description

        high = get_preset("high")
        assert "專業工作" in high.description

        ultra = get_preset("ultra")
        assert "極致效果" in ultra.description

    def test_invalid_preset_level(self) -> None:
        """測試無效的預設等級"""
        with pytest.raises(ValueError, match="not a valid PresetLevel"):
            get_preset("invalid_level")  # type: ignore[arg-type]

    def test_list_presets(self) -> None:
        """測試列出所有預設"""
        presets = list_presets()

        assert "balanced" in presets
        assert "high" in presets
        assert "ultra" in presets
        assert "green_screen" in presets
        assert len(presets) == 4  # noqa: PLR2004

        # 檢查描述包含關鍵資訊
        assert "0.6" in presets["balanced"]
        assert "1024px" in presets["balanced"]

        assert "0.75" in presets["high"]
        assert "1536px" in presets["high"]

        assert "0.95" in presets["ultra"]
        assert "2048px" in presets["ultra"]


class TestPresetConversion:
    """測試預設配置轉換"""

    def test_to_alpha_config(self) -> None:
        """測試轉換為 AlphaConfig"""
        preset = get_preset("balanced")
        alpha_config = preset.to_alpha_config()

        assert alpha_config.mode == AlphaMode.STRAIGHT
        assert alpha_config.edge_decontamination == preset.edge_decontamination
        assert alpha_config.decontamination_strength == preset.decontamination_strength

    def test_to_resolution_config(self) -> None:
        """測試轉換為 ResolutionConfig"""
        balanced = get_preset("balanced")
        res_config = balanced.to_resolution_config()
        assert res_config.mode == ResolutionMode.FIXED_1024

        high = get_preset("high")
        res_config = high.to_resolution_config()
        assert res_config.mode == ResolutionMode.FIXED_1536

        ultra = get_preset("ultra")
        res_config = ultra.to_resolution_config()
        assert res_config.mode == ResolutionMode.FIXED_2048


class TestUltraBackendFromPreset:
    """測試從預設創建 UltraBackend"""

    def test_create_from_balanced_preset(self) -> None:
        """測試從 Balanced 預設創建後端"""
        backend = UltraBackend.from_preset("balanced")

        assert backend.strength == 0.6
        assert backend.resolution_config.mode == ResolutionMode.FIXED_1024
        assert backend.use_trimap_refine is True
        assert backend.alpha_config.edge_decontamination is True

    def test_create_from_high_preset(self) -> None:
        """測試從 High 預設創建後端"""
        backend = UltraBackend.from_preset("high")

        assert backend.strength == 0.75
        assert backend.resolution_config.mode == ResolutionMode.FIXED_1536

    def test_create_from_ultra_preset(self) -> None:
        """測試從 Ultra 預設創建後端"""
        backend = UltraBackend.from_preset("ultra")

        assert backend.strength == 0.95
        assert backend.resolution_config.mode == ResolutionMode.FIXED_2048

    def test_create_with_preset_level_enum(self) -> None:
        """測試使用 PresetLevel 枚舉創建後端"""
        backend = UltraBackend.from_preset(PresetLevel.HIGH)

        assert backend.strength == 0.75
        assert backend.resolution_config.mode == ResolutionMode.FIXED_1536

    def test_create_with_color_filter_override(self) -> None:
        """測試覆蓋色彩過濾設定"""
        green_filter = ColorFilterConfig(color=ColorFilter.GREEN, enabled=True)

        backend = UltraBackend.from_preset("balanced", color_filter=green_filter)

        assert backend.color_filter.color == ColorFilter.GREEN
        assert backend.color_filter.enabled is True

    def test_create_with_device_override(self) -> None:
        """測試覆蓋設備設定"""
        backend = UltraBackend.from_preset("balanced", device="cpu")

        assert backend.device.type == "cpu"

    def test_get_available_presets_from_backend(self) -> None:
        """測試從 UltraBackend 獲取可用預設"""
        presets = UltraBackend.get_available_presets()

        assert isinstance(presets, dict)
        assert "balanced" in presets
        assert "high" in presets
        assert "ultra" in presets


class TestPresetStrengthProgression:
    """測試預設強度的遞增性"""

    def test_strength_increases_across_presets(self) -> None:
        """測試強度在預設之間遞增"""
        balanced = get_preset("balanced")
        high = get_preset("high")
        ultra = get_preset("ultra")

        assert balanced.strength < high.strength < ultra.strength

    def test_decontamination_strength_increases(self) -> None:
        """測試去污染強度遞增"""
        balanced = get_preset("balanced")
        high = get_preset("high")
        ultra = get_preset("ultra")

        assert (
            balanced.decontamination_strength
            < high.decontamination_strength
            < ultra.decontamination_strength
        )


class TestPresetFeatureEnablement:
    """測試預設功能啟用狀態"""

    def test_trimap_enabled_in_all_presets(self) -> None:
        """測試所有預設都啟用 Trimap"""
        for level in ["balanced", "high", "ultra"]:
            preset = get_preset(level)
            assert preset.use_trimap_refine is True

    def test_edge_decontamination_enabled_in_all_presets(self) -> None:
        """測試所有預設都啟用邊緣去污染"""
        for level in ["balanced", "high", "ultra"]:
            preset = get_preset(level)
            assert preset.edge_decontamination is True


class TestPresetColorFilter:
    """測試預設的色彩過濾配置"""

    def test_default_color_filter_is_none(self) -> None:
        """測試預設的色彩過濾為 None"""
        for level in ["balanced", "high", "ultra"]:
            preset = get_preset(level)
            assert preset.color_filter is None

    def test_can_override_with_color_filter(self) -> None:
        """測試可以覆蓋色彩過濾"""
        black_filter = ColorFilterConfig(color=ColorFilter.BLACK, enabled=True)
        backend = UltraBackend.from_preset("high", color_filter=black_filter)

        assert backend.color_filter.color == ColorFilter.BLACK
        assert backend.color_filter.enabled is True


@pytest.mark.parametrize(
    "preset_name,expected_strength,expected_resolution",
    [
        ("balanced", 0.6, "1024"),
        ("high", 0.75, "1536"),
        ("ultra", 0.95, "2048"),
    ],
)
def test_preset_parametrized(
    preset_name: str, expected_strength: float, expected_resolution: str
) -> None:
    """參數化測試所有預設"""
    backend = UltraBackend.from_preset(preset_name)

    assert backend.strength == expected_strength
    assert backend.resolution_config.mode.value == expected_resolution
