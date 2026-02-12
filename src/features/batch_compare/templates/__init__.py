"""
批次比對報告的 HTML/CSS/JS 模板

將模板字串抽離以保持主要程式碼簡潔
"""

from src.features.batch_compare.templates.css import CSS_STYLES
from src.features.batch_compare.templates.html import HTML_BODY
from src.features.batch_compare.templates.js import get_javascript


__all__ = ["CSS_STYLES", "HTML_BODY", "get_javascript"]
