"""
幾何工具模組

提供邊界框和幾何運算功能
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BBox:
    """
    軸對齊邊界框 (Axis-Aligned Bounding Box)

    right 和 bottom 為不包含的座標（exclusive coordinates）

    Attributes:
        left: 左邊界
        top: 上邊界
        right: 右邊界 (不包含)
        bottom: 下邊界 (不包含)
    """

    left: int
    top: int
    right: int
    bottom: int

    def width(self) -> int:
        """計算寬度"""
        return self.right - self.left

    def height(self) -> int:
        """計算高度"""
        return self.bottom - self.top

    def area(self) -> int:
        """計算面積"""
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return 0
        return w * h

    def expand(self, pad: int, *, max_width: int, max_height: int) -> "BBox":
        """
        擴展邊界框

        Args:
            pad: 擴展像素數
            max_width: 最大寬度限制
            max_height: 最大高度限制

        Returns:
            擴展後的邊界框
        """
        if pad <= 0:
            return self
        return BBox(
            left=max(0, self.left - pad),
            top=max(0, self.top - pad),
            right=min(max_width, self.right + pad),
            bottom=min(max_height, self.bottom + pad),
        )

    def overlaps(self, other: "BBox") -> bool:
        """
        檢查是否與另一個邊界框重疊

        Args:
            other: 另一個邊界框

        Returns:
            是否重疊
        """
        return (
            self.left < other.right
            and self.right > other.left
            and self.top < other.bottom
            and self.bottom > other.top
        )

    def union(self, other: "BBox") -> "BBox":
        """
        計算與另一個邊界框的聯集

        Args:
            other: 另一個邊界框

        Returns:
            聯集邊界框
        """
        return BBox(
            left=min(self.left, other.left),
            top=min(self.top, other.top),
            right=max(self.right, other.right),
            bottom=max(self.bottom, other.bottom),
        )
