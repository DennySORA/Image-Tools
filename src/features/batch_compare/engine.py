"""
批次比對引擎

負責執行多組配置的批次測試和品質分析
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.backends.registry import BackendRegistry
from src.common import ColorFilter, ColorFilterConfig
from src.common.preset_config import PresetLevel


logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """品質分析指標"""

    green_spill_ratio: float = 0.0
    green_spill_max: float = 0.0
    green_spill_mean: float = 0.0
    green_spill_score: float = 100.0

    edge_bleeding_ratio: float = 0.0
    edge_bleeding_score: float = 100.0

    over_removal_ratio: float = 0.0
    over_removal_score: float = 100.0

    edge_sharpness: float = 0.0
    alpha_noise: float = 0.0
    alpha_score: float = 100.0

    total_score: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "green_spill": {
                "ratio": self.green_spill_ratio,
                "max_excess": self.green_spill_max,
                "mean_excess": self.green_spill_mean,
                "score": self.green_spill_score,
            },
            "edge_bleeding": {
                "ratio": self.edge_bleeding_ratio,
                "score": self.edge_bleeding_score,
            },
            "over_removal": {
                "ratio": self.over_removal_ratio,
                "score": self.over_removal_score,
            },
            "alpha_quality": {
                "edge_sharpness": self.edge_sharpness,
                "noise": self.alpha_noise,
                "score": self.alpha_score,
            },
            "total_score": self.total_score,
        }


@dataclass
class TestConfig:
    """測試配置"""

    name: str
    description: str
    preset: PresetLevel | None = None
    color_filter: ColorFilterConfig | None = None
    custom: dict[str, Any] | None = None


@dataclass
class ImageResult:
    """單張圖片的測試結果"""

    image_name: str
    metrics: QualityMetrics
    success: bool = True
    error: str | None = None


@dataclass
class ConfigResult:
    """單個配置的測試結果"""

    config_name: str
    description: str
    images: dict[str, ImageResult] = field(default_factory=dict)
    average_score: float = 0.0


@dataclass
class BatchResult:
    """批次測試結果"""

    test_images: list[str]
    configs: dict[str, ConfigResult] = field(default_factory=dict)
    best_config: str = ""


# 預設測試配置
DEFAULT_TEST_CONFIGS: list[TestConfig] = [
    TestConfig(
        name="preset_green_screen",
        description="綠幕預設 (strength=0.8, green despill)",
        preset=PresetLevel.GREEN_SCREEN,
    ),
    TestConfig(
        name="preset_balanced",
        description="平衡預設 (strength=0.6, 1024px)",
        preset=PresetLevel.BALANCED,
        color_filter=ColorFilterConfig(enabled=True, color=ColorFilter.GREEN),
    ),
    TestConfig(
        name="preset_high",
        description="高品質預設 (strength=0.75, 1536px)",
        preset=PresetLevel.HIGH,
        color_filter=ColorFilterConfig(enabled=True, color=ColorFilter.GREEN),
    ),
    TestConfig(
        name="preset_ultra",
        description="極致預設 (strength=0.95, 2048px)",
        preset=PresetLevel.ULTRA,
        color_filter=ColorFilterConfig(enabled=True, color=ColorFilter.GREEN),
    ),
    TestConfig(
        name="custom_aggressive",
        description="自訂激進 (strength=1.0, green, high decontam)",
        custom={
            "strength": 1.0,
            "color_filter": ColorFilterConfig(
                enabled=True,
                color=ColorFilter.GREEN,
                edge_refine_strength=1.0,
            ),
            "use_trimap_refine": True,
        },
    ),
]


class BatchCompareEngine:
    """
    批次比對引擎

    執行多組配置的批次測試，分析處理品質，生成比對報告
    """

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        configs: list[TestConfig] | None = None,
        progress_callback: Any | None = None,
    ) -> None:
        """
        初始化引擎

        Args:
            configs: 測試配置列表，None 使用預設配置
            progress_callback: 進度回調函數
        """
        self.configs = configs or DEFAULT_TEST_CONFIGS
        self.progress_callback = progress_callback

    def run(self, input_dir: Path, output_dir: Path) -> BatchResult:
        """
        執行批次測試

        Args:
            input_dir: 輸入圖片目錄
            output_dir: 輸出目錄

        Returns:
            批次測試結果
        """
        # 掃描測試圖片
        image_files = self._scan_images(input_dir)
        logger.info("找到 %d 張測試圖片", len(image_files))

        result = BatchResult(test_images=[f.name for f in image_files])

        # 測試每個配置
        for config in self.configs:
            logger.info("測試配置: %s", config.name)
            config_result = self._test_config(
                config, image_files, input_dir, output_dir
            )
            result.configs[config.name] = config_result

        # 找出最佳配置
        if result.configs:
            result.best_config = max(
                result.configs.items(),
                key=lambda x: x[1].average_score,
            )[0]

        return result

    def _scan_images(self, input_dir: Path) -> list[Path]:
        """掃描支援格式的圖片"""
        images: list[Path] = []
        for ext in self.SUPPORTED_FORMATS:
            images.extend(input_dir.glob(f"*{ext}"))
            images.extend(input_dir.glob(f"*{ext.upper()}"))
        return sorted(images)

    def _test_config(
        self,
        config: TestConfig,
        image_files: list[Path],
        input_dir: Path,  # noqa: ARG002
        output_dir: Path,
    ) -> ConfigResult:
        """測試單個配置"""
        # 建立後端
        backend = self._create_backend(config)

        # 建立輸出目錄
        config_output = output_dir / config.name
        config_output.mkdir(parents=True, exist_ok=True)

        result = ConfigResult(
            config_name=config.name,
            description=config.description,
        )

        scores = []

        for image_path in image_files:
            output_path = config_output / image_path.name
            logger.info("處理: %s", image_path.name)

            # 執行處理
            success = backend.process(image_path, output_path)

            if success and output_path.exists():
                # 分析品質
                metrics = self._analyze_quality(image_path, output_path)
                result.images[image_path.name] = ImageResult(
                    image_name=image_path.name,
                    metrics=metrics,
                    success=True,
                )
                scores.append(metrics.total_score)
                logger.info("  總分: %.1f", metrics.total_score)
            else:
                result.images[image_path.name] = ImageResult(
                    image_name=image_path.name,
                    metrics=QualityMetrics(),
                    success=False,
                    error="處理失敗",
                )
                logger.error("  處理失敗!")

            # 進度回調
            if self.progress_callback:
                self.progress_callback(config.name, image_path.name)

        # 計算平均分數
        result.average_score = sum(scores) / len(scores) if scores else 0.0

        return result

    def _create_backend(self, config: TestConfig) -> Any:
        """根據配置建立後端"""
        UltraBackend: Any = BackendRegistry.get("ultra")  # noqa: N806

        if config.preset is not None:
            return UltraBackend.from_preset(
                config.preset,
                color_filter=config.color_filter,
            )
        if config.custom is not None:
            return UltraBackend(
                strength=config.custom.get("strength", 0.8),
                color_filter=config.custom.get("color_filter"),
                use_trimap_refine=config.custom.get("use_trimap_refine", True),
            )
        return UltraBackend()

    def _analyze_quality(  # noqa: PLR0915
        self,
        original_path: Path,
        result_path: Path,
    ) -> QualityMetrics:
        """
        分析處理後圖片的品質

        分析項目：
        1. 綠幕殘留 (Green Spill)
        2. 邊緣溢出 (Edge Bleeding)
        3. 過度移除 (Over-removal)
        4. Alpha 品質
        """
        metrics = QualityMetrics()

        # 載入圖片
        original = cv2.imread(str(original_path))
        if original is None:
            return metrics

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        result = cv2.imread(str(result_path), cv2.IMREAD_UNCHANGED)
        if result is None:
            return metrics

        # 分離 RGBA
        if result.shape[2] == 4:  # noqa: PLR2004
            result_rgb = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2RGB)
            alpha = result[:, :, 3]
        else:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            alpha = np.ones(result.shape[:2], dtype=np.uint8) * 255

        alpha_norm = alpha.astype(np.float32) / 255.0

        # === 1. 綠幕殘留分析 ===
        foreground_mask = alpha_norm > 0.5  # noqa: PLR2004
        if np.any(foreground_mask):
            r = result_rgb[:, :, 0].astype(np.float32)
            g = result_rgb[:, :, 1].astype(np.float32)
            b = result_rgb[:, :, 2].astype(np.float32)

            green_excess = g - np.maximum(r, b)
            green_excess_fg = green_excess[foreground_mask]

            green_spill_pixels = np.sum(green_excess_fg > 10)  # noqa: PLR2004
            metrics.green_spill_ratio = float(
                green_spill_pixels / np.sum(foreground_mask)
            )
            metrics.green_spill_max = (
                float(np.max(green_excess_fg)) if len(green_excess_fg) > 0 else 0.0
            )
            metrics.green_spill_mean = (
                float(np.mean(green_excess_fg[green_excess_fg > 0]))
                if np.any(green_excess_fg > 0)
                else 0.0
            )

        # === 2. 邊緣溢出分析 ===
        edge_mask = (alpha_norm > 0.01) & (alpha_norm < 0.99)  # noqa: PLR2004
        if np.any(edge_mask):
            r = result_rgb[:, :, 0].astype(np.float32)
            g = result_rgb[:, :, 1].astype(np.float32)
            b = result_rgb[:, :, 2].astype(np.float32)
            green_excess = g - np.maximum(r, b)

            edge_green_excess = green_excess[edge_mask]
            edge_bleeding_pixels = np.sum(edge_green_excess > 15)  # noqa: PLR2004
            metrics.edge_bleeding_ratio = float(
                edge_bleeding_pixels / np.sum(edge_mask)
            )

        # === 3. 過度移除分析 ===
        orig_r = original_rgb[:, :, 0].astype(np.float32)
        orig_g = original_rgb[:, :, 1].astype(np.float32)
        orig_b = original_rgb[:, :, 2].astype(np.float32)

        non_green_orig = orig_g < np.maximum(orig_r, orig_b) + 30
        removed_mask = alpha_norm < 0.5  # noqa: PLR2004

        over_removal_pixels = np.sum(non_green_orig & removed_mask)
        if np.sum(non_green_orig) > 0:
            metrics.over_removal_ratio = float(
                over_removal_pixels / np.sum(non_green_orig)
            )

        # === 4. Alpha 品質分析 ===
        grad_x = cv2.Sobel(alpha_norm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha_norm, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        if np.any(edge_mask):
            metrics.edge_sharpness = float(np.mean(gradient_magnitude[edge_mask]))

        non_edge = ~edge_mask & (alpha_norm > 0.1) & (alpha_norm < 0.9)  # noqa: PLR2004
        if np.any(non_edge):
            metrics.alpha_noise = float(np.mean(gradient_magnitude[non_edge]))

        # === 5. 計算分數 ===
        metrics.green_spill_score = max(0, 100 - metrics.green_spill_ratio * 1000)
        metrics.edge_bleeding_score = max(0, 100 - metrics.edge_bleeding_ratio * 500)
        metrics.over_removal_score = max(0, 100 - metrics.over_removal_ratio * 200)
        metrics.alpha_score = min(
            100, metrics.edge_sharpness * 500 + (1 - metrics.alpha_noise) * 50
        )

        # 總分（加權平均）
        metrics.total_score = (
            metrics.green_spill_score * 0.4
            + metrics.edge_bleeding_score * 0.3
            + metrics.over_removal_score * 0.2
            + metrics.alpha_score * 0.1
        )

        return metrics
