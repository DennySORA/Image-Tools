"""
比對報告生成器

生成 HTML 視覺化比對報告
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


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

        html_content = self._get_html_template(
            test_images=result.test_images,
            configs=configs_json,
            best_config=result.best_config,
            input_rel_path=str(rel_input),
        )

        html_path = self.output_dir / "compare.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    def _get_html_template(
        self,
        test_images: list[str],
        configs: dict[str, Any],
        best_config: str,
        input_rel_path: str,
    ) -> str:
        """獲取 HTML 模板"""
        return f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BG Removal Compare | 背景移除效果比對</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0a0b;
            --bg-secondary: #141416;
            --bg-tertiary: #1c1c1f;
            --bg-elevated: #232326;
            --text-primary: #f5f5f7;
            --text-secondary: #a1a1a6;
            --text-muted: #6e6e73;
            --accent-cyan: #00d4ff;
            --accent-green: #32d74b;
            --accent-orange: #ff9f0a;
            --accent-red: #ff453a;
            --accent-purple: #bf5af2;
            --border-color: #2d2d30;
            --checker-light: #3a3a3d;
            --checker-dark: #2a2a2d;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;
            --font-mono: 'JetBrains Mono', monospace;
            --font-sans: 'Noto Sans TC', -apple-system, BlinkMacSystemFont, sans-serif;
            --shadow-lg: 0 24px 48px rgba(0, 0, 0, 0.4);
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-smooth: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: var(--font-sans);
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }}

        .header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
            background: rgba(10, 10, 11, 0.85);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-color);
            padding: 0 24px;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .logo {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .logo-icon {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-mono);
            font-weight: 700;
            font-size: 14px;
        }}

        .logo-text {{
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 14px;
            letter-spacing: -0.5px;
        }}

        .logo-text span {{
            color: var(--text-muted);
            font-weight: 400;
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .kbd {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 24px;
            height: 22px;
            padding: 0 6px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
        }}

        .main {{
            padding-top: 56px;
            display: grid;
            grid-template-columns: 280px 1fr;
            min-height: 100vh;
        }}

        .sidebar {{
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            padding: 20px 16px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            overflow-y: auto;
            height: calc(100vh - 56px);
            position: sticky;
            top: 56px;
        }}

        .section-title {{
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}

        .image-list {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .image-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 12px;
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all var(--transition-fast);
            border: 1px solid transparent;
        }}

        .image-item:hover {{
            background: var(--bg-tertiary);
        }}

        .image-item.active {{
            background: var(--bg-elevated);
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.2);
        }}

        .image-thumb {{
            width: 40px;
            height: 40px;
            border-radius: var(--radius-sm);
            object-fit: cover;
            background: var(--checker-dark);
            background-image:
                linear-gradient(45deg, var(--checker-light) 25%, transparent 25%),
                linear-gradient(-45deg, var(--checker-light) 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, var(--checker-light) 75%),
                linear-gradient(-45deg, transparent 75%, var(--checker-light) 75%);
            background-size: 8px 8px;
            background-position: 0 0, 0 4px, 4px -4px, -4px 0px;
        }}

        .image-info {{
            flex: 1;
            min-width: 0;
        }}

        .image-name {{
            font-size: 13px;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .image-score {{
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
        }}

        .config-list {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .config-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px;
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all var(--transition-fast);
            border: 1px solid transparent;
            background: var(--bg-tertiary);
        }}

        .config-item:hover {{
            background: var(--bg-elevated);
        }}

        .config-item.active {{
            border-color: var(--accent-green);
            box-shadow: 0 0 0 1px rgba(50, 215, 75, 0.2);
        }}

        .config-item.active .config-name {{
            color: var(--accent-green);
        }}

        .config-info {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .config-name {{
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            transition: color var(--transition-fast);
        }}

        .config-desc {{
            font-size: 11px;
            color: var(--text-muted);
        }}

        .config-score {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .score-badge {{
            font-family: var(--font-mono);
            font-size: 13px;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: var(--radius-sm);
            background: rgba(50, 215, 75, 0.15);
            color: var(--accent-green);
        }}

        .score-badge.warning {{
            background: rgba(255, 159, 10, 0.15);
            color: var(--accent-orange);
        }}

        .score-badge.danger {{
            background: rgba(255, 69, 58, 0.15);
            color: var(--accent-red);
        }}

        .stats-panel {{
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
            padding: 16px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}

        .stat-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .stat-label {{
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-value {{
            font-family: var(--font-mono);
            font-size: 18px;
            font-weight: 700;
        }}

        .stat-value.green {{ color: var(--accent-green); }}
        .stat-value.orange {{ color: var(--accent-orange); }}
        .stat-value.red {{ color: var(--accent-red); }}
        .stat-value.cyan {{ color: var(--accent-cyan); }}

        .stat-bar {{
            height: 4px;
            background: var(--bg-primary);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 4px;
        }}

        .stat-bar-fill {{
            height: 100%;
            border-radius: 2px;
            transition: width var(--transition-smooth);
        }}

        .stat-bar-fill.green {{ background: var(--accent-green); }}
        .stat-bar-fill.orange {{ background: var(--accent-orange); }}
        .stat-bar-fill.red {{ background: var(--accent-red); }}
        .stat-bar-fill.cyan {{ background: var(--accent-cyan); }}

        .compare-area {{
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        .compare-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .compare-title {{
            font-family: var(--font-mono);
            font-size: 13px;
            color: var(--text-secondary);
        }}

        .compare-title strong {{
            color: var(--text-primary);
            font-weight: 600;
        }}

        .view-toggle {{
            display: flex;
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            padding: 3px;
            gap: 2px;
        }}

        .view-btn {{
            padding: 6px 14px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 500;
            border-radius: 4px;
            cursor: pointer;
            transition: all var(--transition-fast);
        }}

        .view-btn:hover {{
            color: var(--text-primary);
        }}

        .view-btn.active {{
            background: var(--bg-elevated);
            color: var(--text-primary);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}

        .compare-container {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-secondary);
            border-radius: var(--radius-lg);
            overflow: hidden;
            position: relative;
            min-height: 500px;
        }}

        .checker-bg {{
            position: absolute;
            inset: 0;
            background-color: var(--checker-dark);
            background-image:
                linear-gradient(45deg, var(--checker-light) 25%, transparent 25%),
                linear-gradient(-45deg, var(--checker-light) 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, var(--checker-light) 75%),
                linear-gradient(-45deg, transparent 75%, var(--checker-light) 75%);
            background-size: 16px 16px;
            background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
        }}

        .slider-compare {{
            position: relative;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .slider-wrapper {{
            position: relative;
            max-width: 100%;
            max-height: 100%;
        }}

        .slider-image {{
            display: block;
            max-width: 100%;
            max-height: calc(100vh - 200px);
            object-fit: contain;
        }}

        .slider-image.original {{
            position: absolute;
            top: 0;
            left: 0;
            clip-path: inset(0 0 0 0);
        }}

        .slider-image.processed {{
            position: relative;
            z-index: 1;
        }}

        .slider-divider {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 3px;
            background: var(--text-primary);
            cursor: ew-resize;
            z-index: 10;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }}

        .slider-divider::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 40px;
            height: 40px;
            background: var(--bg-elevated);
            border: 2px solid var(--text-primary);
            border-radius: 50%;
            box-shadow: var(--shadow-lg);
        }}

        .slider-divider::after {{
            content: '\\2194';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 16px;
            color: var(--text-primary);
            z-index: 1;
        }}

        .slider-label {{
            position: absolute;
            bottom: 16px;
            padding: 6px 12px;
            background: rgba(10, 10, 11, 0.85);
            backdrop-filter: blur(10px);
            border-radius: var(--radius-sm);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            z-index: 5;
        }}

        .slider-label.left {{
            left: 16px;
            color: var(--accent-orange);
        }}

        .slider-label.right {{
            right: 16px;
            color: var(--accent-green);
        }}

        .side-by-side {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            width: 100%;
            height: 100%;
            padding: 4px;
        }}

        .side-panel {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border-radius: var(--radius-md);
        }}

        .side-panel img {{
            max-width: 100%;
            max-height: calc(100vh - 200px);
            object-fit: contain;
        }}

        .side-panel .panel-label {{
            position: absolute;
            top: 12px;
            left: 12px;
            padding: 6px 12px;
            background: rgba(10, 10, 11, 0.85);
            backdrop-filter: blur(10px);
            border-radius: var(--radius-sm);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        .side-panel.original .panel-label {{
            color: var(--accent-orange);
        }}

        .side-panel.processed .panel-label {{
            color: var(--accent-green);
        }}

        .hidden {{
            display: none !important;
        }}

        .keyboard-hints {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 12px;
            padding: 12px 16px;
            background: rgba(20, 20, 22, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            font-size: 12px;
            color: var(--text-secondary);
            z-index: 50;
        }}

        .hint-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .loading {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 13px;
        }}

        .loading::before {{
            content: '';
            width: 20px;
            height: 20px;
            border: 2px solid var(--border-color);
            border-top-color: var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        @media (max-width: 900px) {{
            .main {{
                grid-template-columns: 1fr;
            }}

            .sidebar {{
                display: none;
            }}

            .compare-area {{
                padding: 16px;
            }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">
            <div class="logo-icon">BG</div>
            <div class="logo-text">Compare <span>v1.0</span></div>
        </div>
        <div class="header-actions">
            <span style="font-size: 12px; color: var(--text-muted);">快捷鍵：</span>
            <kbd>\\u2190</kbd><kbd>\\u2192</kbd> 切換圖片
            <kbd>1</kbd>-<kbd>5</kbd> 切換配置
        </div>
    </header>

    <main class="main">
        <aside class="sidebar">
            <section>
                <h3 class="section-title">測試圖片</h3>
                <div class="image-list" id="imageList"></div>
            </section>

            <section>
                <h3 class="section-title">處理配置</h3>
                <div class="config-list" id="configList"></div>
            </section>

            <section>
                <h3 class="section-title">品質分析</h3>
                <div class="stats-panel">
                    <div class="stats-grid" id="statsGrid"></div>
                </div>
            </section>
        </aside>

        <div class="compare-area">
            <div class="compare-header">
                <div class="compare-title">
                    <strong id="currentImageName">選擇圖片</strong>
                    <span id="currentConfigName"></span>
                </div>
                <div class="view-toggle">
                    <button class="view-btn active" data-view="slider">滑動比對</button>
                    <button class="view-btn" data-view="sidebyside">並排顯示</button>
                </div>
            </div>

            <div class="compare-container">
                <div class="checker-bg"></div>

                <div class="slider-compare" id="sliderView">
                    <div class="slider-wrapper" id="sliderWrapper">
                        <img class="slider-image processed" id="processedImage" src="" alt="Processed">
                        <img class="slider-image original" id="originalImage" src="" alt="Original">
                        <div class="slider-divider" id="sliderDivider"></div>
                    </div>
                    <div class="slider-label left">原圖</div>
                    <div class="slider-label right">處理後</div>
                </div>

                <div class="side-by-side hidden" id="sideBySideView">
                    <div class="side-panel original">
                        <div class="checker-bg"></div>
                        <img id="sideOriginal" src="" alt="Original">
                        <div class="panel-label">原圖</div>
                    </div>
                    <div class="side-panel processed">
                        <div class="checker-bg"></div>
                        <img id="sideProcessed" src="" alt="Processed">
                        <div class="panel-label">處理後</div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <div class="keyboard-hints">
        <div class="hint-item"><kbd>V</kbd> 切換視圖</div>
        <div class="hint-item"><kbd>Space</kbd> 顯示/隱藏原圖</div>
    </div>

    <script>
        const reportData = {{
            test_images: {json.dumps(test_images)},
            configs: {json.dumps(configs)},
            best_config: "{best_config}"
        }};
        const inputRelPath = "{input_rel_path}";

        let fullReport = null;
        fetch('test_report.json')
            .then(r => r.json())
            .then(data => {{
                fullReport = data;
                updateStats();
            }})
            .catch(e => console.log('Could not load detailed report'));

        let currentImage = reportData.test_images[0];
        let currentConfig = reportData.best_config || Object.keys(reportData.configs)[0];
        let currentView = 'slider';
        let sliderPosition = 50;

        const imageList = document.getElementById('imageList');
        const configList = document.getElementById('configList');
        const statsGrid = document.getElementById('statsGrid');
        const currentImageName = document.getElementById('currentImageName');
        const currentConfigName = document.getElementById('currentConfigName');
        const sliderView = document.getElementById('sliderView');
        const sideBySideView = document.getElementById('sideBySideView');
        const originalImage = document.getElementById('originalImage');
        const processedImage = document.getElementById('processedImage');
        const sideOriginal = document.getElementById('sideOriginal');
        const sideProcessed = document.getElementById('sideProcessed');
        const sliderDivider = document.getElementById('sliderDivider');
        const sliderWrapper = document.getElementById('sliderWrapper');

        function init() {{
            renderImageList();
            renderConfigList();
            updateImages();
            setupSlider();
            setupKeyboard();
            setupViewToggle();
        }}

        function renderImageList() {{
            imageList.innerHTML = reportData.test_images.map((img, i) => {{
                const shortName = img.replace('Gemini_Generated_Image_', '').replace('.png', '');
                const displayName = shortName.length > 15 ? shortName.slice(0, 12) + '...' : shortName;
                return `
                    <div class="image-item ${{img === currentImage ? 'active' : ''}}" data-image="${{img}}">
                        <img class="image-thumb" src="${{inputRelPath}}/${{img}}" alt="${{shortName}}" onerror="this.style.display='none'">
                        <div class="image-info">
                            <div class="image-name">${{displayName}}</div>
                            <div class="image-score">#${{i + 1}}</div>
                        </div>
                    </div>
                `;
            }}).join('');

            imageList.querySelectorAll('.image-item').forEach(item => {{
                item.addEventListener('click', () => {{
                    currentImage = item.dataset.image;
                    updateSelection();
                }});
            }});
        }}

        function renderConfigList() {{
            const configs = Object.entries(reportData.configs);
            configList.innerHTML = configs.map(([key, config], i) => {{
                const scoreClass = config.average_score >= 95 ? '' : config.average_score >= 90 ? 'warning' : 'danger';
                const isBest = key === reportData.best_config;
                return `
                    <div class="config-item ${{key === currentConfig ? 'active' : ''}}" data-config="${{key}}">
                        <div class="config-info">
                            <div class="config-name">${{isBest ? '\\uD83C\\uDFC6 ' : ''}}${{key}}</div>
                            <div class="config-desc">${{config.description}}</div>
                        </div>
                        <div class="config-score">
                            <span class="score-badge ${{scoreClass}}">${{config.average_score.toFixed(1)}}</span>
                        </div>
                    </div>
                `;
            }}).join('');

            configList.querySelectorAll('.config-item').forEach(item => {{
                item.addEventListener('click', () => {{
                    currentConfig = item.dataset.config;
                    updateSelection();
                }});
            }});
        }}

        function updateSelection() {{
            imageList.querySelectorAll('.image-item').forEach(item => {{
                item.classList.toggle('active', item.dataset.image === currentImage);
            }});

            configList.querySelectorAll('.config-item').forEach(item => {{
                item.classList.toggle('active', item.dataset.config === currentConfig);
            }});

            updateImages();
            updateStats();
        }}

        function updateImages() {{
            const originalPath = `${{inputRelPath}}/${{currentImage}}`;
            const processedPath = `${{currentConfig}}/${{currentImage}}`;

            originalImage.src = originalPath;
            processedImage.src = processedPath;
            sideOriginal.src = originalPath;
            sideProcessed.src = processedPath;

            const shortName = currentImage.replace('Gemini_Generated_Image_', '').replace('.png', '');
            currentImageName.textContent = shortName;
            currentConfigName.textContent = ` \\u2192 ${{currentConfig}}`;
        }}

        function updateStats() {{
            if (!fullReport) {{
                statsGrid.innerHTML = '<div class="loading">載入中...</div>';
                return;
            }}

            const configData = fullReport.configs[currentConfig];
            const imageData = configData?.images?.[currentImage];

            if (!imageData) {{
                statsGrid.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">無數據</div>';
                return;
            }}

            const stats = [
                {{ label: '綠幕殘留', value: imageData.green_spill.score, color: imageData.green_spill.score >= 90 ? 'green' : 'orange' }},
                {{ label: '邊緣溢出', value: imageData.edge_bleeding.score, color: imageData.edge_bleeding.score >= 90 ? 'green' : 'orange' }},
                {{ label: '過度移除', value: imageData.over_removal.score, color: imageData.over_removal.score >= 50 ? 'green' : 'red' }},
                {{ label: '總分', value: imageData.total_score, color: 'cyan' }}
            ];

            statsGrid.innerHTML = stats.map(stat => `
                <div class="stat-item">
                    <div class="stat-label">${{stat.label}}</div>
                    <div class="stat-value ${{stat.color}}">${{stat.value.toFixed(1)}}</div>
                    <div class="stat-bar">
                        <div class="stat-bar-fill ${{stat.color}}" style="width: ${{stat.value}}%"></div>
                    </div>
                </div>
            `).join('');
        }}

        function setupSlider() {{
            let isDragging = false;

            const updateSlider = (clientX) => {{
                const rect = sliderWrapper.getBoundingClientRect();
                const x = clientX - rect.left;
                const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
                sliderPosition = percent;

                sliderDivider.style.left = `${{percent}}%`;
                originalImage.style.clipPath = `inset(0 ${{100 - percent}}% 0 0)`;
            }};

            sliderDivider.addEventListener('mousedown', (e) => {{
                isDragging = true;
                e.preventDefault();
            }});

            document.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    updateSlider(e.clientX);
                }}
            }});

            document.addEventListener('mouseup', () => {{
                isDragging = false;
            }});

            sliderWrapper.addEventListener('click', (e) => {{
                if (!isDragging) {{
                    updateSlider(e.clientX);
                }}
            }});

            sliderDivider.style.left = '50%';
            originalImage.style.clipPath = 'inset(0 50% 0 0)';
        }}

        function setupKeyboard() {{
            document.addEventListener('keydown', (e) => {{
                const images = reportData.test_images;
                const configs = Object.keys(reportData.configs);
                const currentImageIndex = images.indexOf(currentImage);
                const currentConfigIndex = configs.indexOf(currentConfig);

                switch(e.key) {{
                    case 'ArrowLeft':
                        if (currentImageIndex > 0) {{
                            currentImage = images[currentImageIndex - 1];
                            updateSelection();
                        }}
                        break;
                    case 'ArrowRight':
                        if (currentImageIndex < images.length - 1) {{
                            currentImage = images[currentImageIndex + 1];
                            updateSelection();
                        }}
                        break;
                    case '1': case '2': case '3': case '4': case '5':
                        const idx = parseInt(e.key) - 1;
                        if (idx < configs.length) {{
                            currentConfig = configs[idx];
                            updateSelection();
                        }}
                        break;
                    case 'v':
                    case 'V':
                        toggleView();
                        break;
                    case ' ':
                        e.preventDefault();
                        if (currentView === 'slider') {{
                            if (originalImage.style.clipPath === 'inset(0 0% 0 0)') {{
                                originalImage.style.clipPath = 'inset(0 50% 0 0)';
                                sliderDivider.style.left = '50%';
                            }} else {{
                                originalImage.style.clipPath = 'inset(0 0% 0 0)';
                                sliderDivider.style.left = '100%';
                            }}
                        }}
                        break;
                }}
            }});
        }}

        function setupViewToggle() {{
            document.querySelectorAll('.view-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const view = btn.dataset.view;
                    setView(view);
                }});
            }});
        }}

        function setView(view) {{
            currentView = view;
            document.querySelectorAll('.view-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.view === view);
            }});

            sliderView.classList.toggle('hidden', view !== 'slider');
            sideBySideView.classList.toggle('hidden', view !== 'sidebyside');
        }}

        function toggleView() {{
            setView(currentView === 'slider' ? 'sidebyside' : 'slider');
        }}

        init();
    </script>
</body>
</html>'''
