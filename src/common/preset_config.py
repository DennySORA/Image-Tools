"""
背景移除預設配置

提供三個簡化的預設配置等級：
- Balanced（中等）：平衡速度與品質，適合日常使用
- High（高）：高品質處理，適合專業工作
- Ultra（最強）：全部參數拉滿，追求極致效果
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from src.common.alpha_config import AlphaConfig, ResolutionConfig, ResolutionMode
from src.common.color_filter import ColorFilterConfig


class PresetLevel(StrEnum):
    """預設配置等級"""

    BALANCED = "balanced"  # 中等：平衡速度與品質
    HIGH = "high"  # 高：高品質處理
    ULTRA = "ultra"  # 最強：全部參數拉滿


class BackgroundRemovalPreset(BaseModel):
    """
    背景移除預設配置

    封裝所有相關參數，提供開箱即用的配置
    """

    # 基本配置
    strength: float = Field(ge=0.1, le=1.0)
    resolution_mode: ResolutionMode

    # Trimap refinement
    use_trimap_refine: bool

    # Portrait matting（人像精修）
    use_portrait_matting: bool
    portrait_matting_strength: float = Field(ge=0.1, le=1.0)
    portrait_matting_model: str = "enhanced"  # "enhanced" 或 "birefnet"

    # Alpha 處理
    edge_decontamination: bool
    decontamination_strength: float = Field(ge=0.0, le=1.0)

    # 色彩過濾（可選，預設關閉）
    color_filter: ColorFilterConfig | None = None

    @property
    def level_name(self) -> str:
        """取得配置等級名稱"""
        if self.strength <= 0.6:  # noqa: PLR2004
            return "Balanced"
        if self.strength <= 0.8:  # noqa: PLR2004
            return "High"
        return "Ultra"

    @property
    def description(self) -> str:
        """取得配置描述"""
        descriptions = {
            "Balanced": "平衡速度與品質，適合日常使用",
            "High": "高品質處理，適合專業工作",
            "Ultra": "全部參數拉滿，追求極致效果",
        }
        return descriptions.get(self.level_name, "自訂配置")

    def to_alpha_config(self) -> AlphaConfig:
        """轉換為 AlphaConfig"""
        return AlphaConfig(
            edge_decontamination=self.edge_decontamination,
            decontamination_strength=self.decontamination_strength,
        )

    def to_resolution_config(self) -> ResolutionConfig:
        """轉換為 ResolutionConfig"""
        return ResolutionConfig(mode=self.resolution_mode)


# 預設配置定義
PRESET_BALANCED = BackgroundRemovalPreset(
    strength=0.6,
    resolution_mode=ResolutionMode.FIXED_1024,
    use_trimap_refine=True,
    use_portrait_matting=False,
    portrait_matting_strength=0.5,
    portrait_matting_model="enhanced",
    edge_decontamination=True,
    decontamination_strength=0.6,
)

PRESET_HIGH = BackgroundRemovalPreset(
    strength=0.75,
    resolution_mode=ResolutionMode.FIXED_1536,
    use_trimap_refine=True,
    use_portrait_matting=True,
    portrait_matting_strength=0.7,
    portrait_matting_model="enhanced",
    edge_decontamination=True,
    decontamination_strength=0.75,
)

PRESET_ULTRA = BackgroundRemovalPreset(
    strength=0.95,
    resolution_mode=ResolutionMode.FIXED_2048,
    use_trimap_refine=True,
    use_portrait_matting=True,
    portrait_matting_strength=0.9,
    portrait_matting_model="birefnet",  # 使用 BiRefNet-matting 專業模型
    edge_decontamination=True,
    decontamination_strength=0.95,
)


def get_preset(level: PresetLevel | str) -> BackgroundRemovalPreset:
    """
    取得預設配置

    Args:
        level: 預設等級（balanced / high / ultra）

    Returns:
        預設配置

    Raises:
        ValueError: 如果等級不存在

    Examples:
        >>> preset = get_preset("balanced")
        >>> preset = get_preset(PresetLevel.HIGH)
    """
    if isinstance(level, str):
        level = PresetLevel(level.lower())

    presets = {
        PresetLevel.BALANCED: PRESET_BALANCED,
        PresetLevel.HIGH: PRESET_HIGH,
        PresetLevel.ULTRA: PRESET_ULTRA,
    }

    if level not in presets:
        msg = f"Unknown preset level: {level}"
        raise ValueError(msg)

    return presets[level]


def list_presets() -> dict[str, str]:
    """
    列出所有可用的預設配置

    Returns:
        預設等級 -> 描述的字典
    """
    return {
        "balanced": "中等 - 平衡速度與品質（強度 0.6，1024px）",
        "high": "高 - 高品質處理（強度 0.75，1536px，人像精修）",
        "ultra": "最強 - 全部參數拉滿（強度 0.95，2048px，MODNet 人像精修）",
    }


def print_preset_comparison() -> None:
    """列印預設配置對比表（用於 CLI 展示）"""
    print("\n" + "=" * 80)
    print("背景移除預設配置對比".center(80))
    print("=" * 80)
    print()

    headers = ["功能", "Balanced（中等）", "High（高）", "Ultra（最強）"]
    rows = [
        ["強度", "0.6", "0.75", "0.95"],
        ["解析度", "1024px", "1536px", "2048px"],
        ["Trimap 精修", "✓", "✓", "✓"],
        ["人像精修", "✗", "✓ (Enhanced)", "✓ (MODNet)"],
        ["邊緣去污染", "✓ (0.6)", "✓ (0.75)", "✓ (0.95)"],
        ["適用場景", "日常使用", "專業工作", "極致品質"],
        ["處理速度", "快", "中", "慢"],
    ]

    # 計算每列寬度
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(4)]

    # 列印表頭
    header_line = " | ".join(
        headers[i].ljust(col_widths[i]) for i in range(len(headers))
    )
    print(header_line)
    print("-" * len(header_line))

    # 列印內容
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))

    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    # 展示預設配置
    print_preset_comparison()

    # 測試取得預設
    for level in PresetLevel:
        preset = get_preset(level)
        print(f"\n{level.value.upper()}:")
        print(f"  描述: {preset.description}")
        print(f"  強度: {preset.strength}")
        print(f"  解析度: {preset.resolution_mode.value}")
        print(f"  人像精修: {preset.use_portrait_matting}")
