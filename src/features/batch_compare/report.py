"""
比對報告生成器

生成 HTML 視覺化比對報告
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.features.batch_compare.templates import (
    CSS_STYLES,
    HTML_BODY,
    get_javascript,
)


if TYPE_CHECKING:
    from src.features.batch_compare.engine import BatchResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    HTML 報告生成器

    生成可互動的視覺化比對報告
    """

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        """
        初始化生成器

        Args:
            input_dir: 原始圖片目錄
            output_dir: 輸出目錄
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

    def generate(self, result: "BatchResult") -> Path:
        """
        生成報告

        Args:
            result: 批次測試結果

        Returns:
            報告文件路徑
        """
        # 生成 JSON 數據
        self._generate_json(result)

        # 生成 HTML 報告
        html_path = self._generate_html(result)

        logger.info("報告已生成: %s", html_path)
        return html_path

    def _generate_json(self, result: "BatchResult") -> Path:
        """生成 JSON 數據文件"""
        data: dict[str, Any] = {
            "test_images": result.test_images,
            "best_config": result.best_config,
            "generated_at": datetime.now().isoformat(),
            "configs": {},
        }

        for config_name, config_result in result.configs.items():
            config_data: dict[str, Any] = {
                "description": config_result.description,
                "average_score": config_result.average_score,
                "images": {},
            }
            for img_name, img_result in config_result.images.items():
                config_data["images"][img_name] = img_result.metrics.to_dict()
            data["configs"][config_name] = config_data

        json_path = self.output_dir / "test_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return json_path

    def _generate_html(self, result: "BatchResult") -> Path:
        """生成 HTML 報告"""
        # 計算相對路徑
        try:
            rel_input = self.input_dir.relative_to(self.output_dir.parent)
        except ValueError:
            rel_input = self.input_dir

        # 構建配置數據
        configs_json = {}
        for config_name, config_result in result.configs.items():
            configs_json[config_name] = {
                "description": config_result.description,
                "average_score": round(config_result.average_score, 2),
            }

        html_content = self._build_html(
            test_images=result.test_images,
            configs=configs_json,
            best_config=result.best_config,
            input_rel_path=str(rel_input),
        )

        html_path = self.output_dir / "compare.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    def _build_html(
        self,
        test_images: list[str],
        configs: dict[str, Any],
        best_config: str,
        input_rel_path: str,
    ) -> str:
        """組裝完整的 HTML 文檔"""
        js_code = get_javascript(
            test_images_json=json.dumps(test_images),
            configs_json=json.dumps(configs),
            best_config=best_config,
            input_rel_path=input_rel_path,
        )

        return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BG Removal Compare | 背景移除效果比對</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
{CSS_STYLES}
    </style>
</head>
<body>
{HTML_BODY}
    <script>
{js_code}
    </script>
</body>
</html>"""
