"""
後端模組

提供各種背景移除後端的實作
"""

from .registry import BackendRegistry
from .rembg import RembgBackend
from .transparent_bg import TransparentBgBackend
from .backgroundremover import BackgroundRemoverBackend
from .greenscreen import GreenScreenBackend

__all__ = [
    'BackendRegistry',
    'RembgBackend',
    'TransparentBgBackend',
    'BackgroundRemoverBackend',
    'GreenScreenBackend',
]
