"""
色彩過濾模組

提供背景顏色過濾功能的共用定義
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ColorFilter(StrEnum):
    """背景顏色過濾選項"""

    NONE = "none"  # 不使用色彩過濾
    BLACK = "black"  # 純黑背景
    WHITE = "white"  # 純白背景
    GREEN = "green"  # 綠幕背景


class ColorFilterConfig(BaseModel):
    """
    色彩過濾設定

    Attributes:
        enabled: 是否啟用色彩過濾
        color: 背景顏色類型
        hue_range: HSV 色相範圍（針對有色背景，如綠幕）
        value_range: 明度範圍（針對黑白背景）
        saturation_min: 飽和度最小值
        edge_refine_strength: 邊緣精細化強度 (0.0-1.0)
    """

    enabled: bool = False
    color: ColorFilter = ColorFilter.NONE
    hue_range: tuple[int, int] = Field(default=(35, 85))
    value_range: tuple[int, int] = Field(default=(0, 30))
    saturation_min: int = Field(default=40)
    edge_refine_strength: float = Field(default=0.8, ge=0.0, le=1.0)
