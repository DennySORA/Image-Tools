"""
批次比對功能模組

提供背景移除參數批次測試和視覺化比對報告生成功能
"""

from src.features.batch_compare.engine import BatchCompareEngine
from src.features.batch_compare.report import ReportGenerator


__all__ = ["BatchCompareEngine", "ReportGenerator"]
