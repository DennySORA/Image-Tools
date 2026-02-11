#!/usr/bin/env python3
"""
批次測試背景移除參數

針對 test_file 目錄中的圖片，測試多組參數組合，
並生成詳細的圖片品質分析報告。

測試組合：
1. green_screen 預設（針對綠幕優化）
2. balanced 預設（平衡）
3. high 預設（高品質）
4. ultra 預設（極致）
5. 自訂組合：高強度 + 綠幕過濾
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common import ColorFilter, ColorFilterConfig
from src.common.preset_config import PresetLevel, get_preset
from src.backends.registry import BackendRegistry

# 確保後端已註冊
import src.backends  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 測試配置
TEST_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "preset_green_screen",
        "description": "綠幕預設 (strength=0.8, green despill)",
        "preset": PresetLevel.GREEN_SCREEN,
    },
    {
        "name": "preset_balanced",
        "description": "平衡預設 (strength=0.6, 1024px)",
        "preset": PresetLevel.BALANCED,
        "color_filter": ColorFilterConfig(enabled=True, color=ColorFilter.GREEN),
    },
    {
        "name": "preset_high",
        "description": "高品質預設 (strength=0.75, 1536px)",
        "preset": PresetLevel.HIGH,
        "color_filter": ColorFilterConfig(enabled=True, color=ColorFilter.GREEN),
    },
    {
        "name": "preset_ultra",
        "description": "極致預設 (strength=0.95, 2048px)",
        "preset": PresetLevel.ULTRA,
        "color_filter": ColorFilterConfig(enabled=True, color=ColorFilter.GREEN),
    },
    {
        "name": "custom_aggressive",
        "description": "自訂激進 (strength=1.0, green, high decontam)",
        "custom": {
            "strength": 1.0,
            "color_filter": ColorFilterConfig(
                enabled=True,
                color=ColorFilter.GREEN,
                edge_refine_strength=1.0,
            ),
            "use_trimap_refine": True,
        },
    },
]


def analyze_image_quality(
    original_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """
    深度分析處理後圖片的品質

    分析項目：
    1. 邊緣溢出 (Edge Bleeding)
    2. 綠幕殘留 (Green Spill)
    3. 過度移除 (Over-removal)
    4. Alpha 品質
    5. 邊緣銳利度

    Returns:
        品質分析報告
    """
    # 載入圖片
    original = cv2.imread(str(original_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    result = cv2.imread(str(result_path), cv2.IMREAD_UNCHANGED)
    if result is None:
        return {"error": "無法載入結果圖片"}

    # 分離 RGBA
    if result.shape[2] == 4:  # noqa: PLR2004
        result_rgb = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = result[:, :, 3]
    else:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        alpha = np.ones(result.shape[:2], dtype=np.uint8) * 255

    alpha_norm = alpha.astype(np.float32) / 255.0

    # === 1. 綠幕殘留分析 ===
    # 在前景區域（alpha > 0.5）檢測綠色過剩
    foreground_mask = alpha_norm > 0.5
    if np.any(foreground_mask):
        r = result_rgb[:, :, 0].astype(np.float32)
        g = result_rgb[:, :, 1].astype(np.float32)
        b = result_rgb[:, :, 2].astype(np.float32)

        # 綠色過剩 = G - max(R, B)
        green_excess = g - np.maximum(r, b)
        green_excess_fg = green_excess[foreground_mask]

        # 統計綠色殘留
        green_spill_pixels = np.sum(green_excess_fg > 10)  # 閾值 10
        green_spill_ratio = green_spill_pixels / np.sum(foreground_mask)
        green_spill_max = np.max(green_excess_fg) if len(green_excess_fg) > 0 else 0
        green_spill_mean = np.mean(green_excess_fg[green_excess_fg > 0]) if np.any(green_excess_fg > 0) else 0
    else:
        green_spill_ratio = 0
        green_spill_max = 0
        green_spill_mean = 0

    # === 2. 邊緣溢出分析 ===
    # 在邊緣區域（0.01 < alpha < 0.99）檢測顏色異常
    edge_mask = (alpha_norm > 0.01) & (alpha_norm < 0.99)
    if np.any(edge_mask):
        edge_green_excess = green_excess[edge_mask]
        edge_bleeding_pixels = np.sum(edge_green_excess > 15)  # 邊緣更嚴格
        edge_bleeding_ratio = edge_bleeding_pixels / np.sum(edge_mask)
    else:
        edge_bleeding_ratio = 0

    # === 3. 過度移除分析 ===
    # 比較原圖非綠色區域是否被錯誤移除
    # 原圖中非綠色區域（綠色不佔優）
    orig_r = original_rgb[:, :, 0].astype(np.float32)
    orig_g = original_rgb[:, :, 1].astype(np.float32)
    orig_b = original_rgb[:, :, 2].astype(np.float32)

    # 非綠色區域 = 綠色不是最大值，或差距小
    non_green_orig = (orig_g < np.maximum(orig_r, orig_b) + 30)

    # 這些區域被移除了（alpha < 0.5）
    removed_mask = alpha_norm < 0.5
    over_removal_pixels = np.sum(non_green_orig & removed_mask)
    over_removal_ratio = over_removal_pixels / np.sum(non_green_orig) if np.sum(non_green_orig) > 0 else 0

    # === 4. Alpha 品質分析 ===
    # Alpha 梯度（邊緣平滑度）
    grad_x = cv2.Sobel(alpha_norm, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha_norm, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 邊緣銳利度（平均梯度）
    edge_sharpness = np.mean(gradient_magnitude[edge_mask]) if np.any(edge_mask) else 0

    # Alpha 噪點（非邊緣區域的梯度）
    non_edge = ~edge_mask & (alpha_norm > 0.1) & (alpha_norm < 0.9)
    alpha_noise = np.mean(gradient_magnitude[non_edge]) if np.any(non_edge) else 0

    # === 5. 整體評分 ===
    # 各項權重
    # 綠幕殘留: 40%（這是主要問題）
    # 邊緣溢出: 30%
    # 過度移除: 20%
    # Alpha 品質: 10%

    green_score = max(0, 100 - green_spill_ratio * 1000)  # 0.1% 殘留 = -100 分
    edge_score = max(0, 100 - edge_bleeding_ratio * 500)
    removal_score = max(0, 100 - over_removal_ratio * 200)
    alpha_score = min(100, edge_sharpness * 500 + (1 - alpha_noise) * 50)

    total_score = (
        green_score * 0.4 +
        edge_score * 0.3 +
        removal_score * 0.2 +
        alpha_score * 0.1
    )

    return {
        "green_spill": {
            "ratio": float(green_spill_ratio),
            "max_excess": float(green_spill_max),
            "mean_excess": float(green_spill_mean),
            "score": float(green_score),
        },
        "edge_bleeding": {
            "ratio": float(edge_bleeding_ratio),
            "score": float(edge_score),
        },
        "over_removal": {
            "ratio": float(over_removal_ratio),
            "score": float(removal_score),
        },
        "alpha_quality": {
            "edge_sharpness": float(edge_sharpness),
            "noise": float(alpha_noise),
            "score": float(alpha_score),
        },
        "total_score": float(total_score),
    }


def run_test(
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    執行完整測試

    Args:
        input_dir: 輸入目錄
        output_dir: 輸出目錄

    Returns:
        完整測試報告
    """
    # 收集測試圖片
    image_files = list(input_dir.glob("*.png"))
    logger.info("找到 %d 張測試圖片", len(image_files))

    results: dict[str, Any] = {
        "test_images": [f.name for f in image_files],
        "configs": {},
    }

    # 載入模型一次
    logger.info("載入 RMBG-2.0 模型...")

    for config in TEST_CONFIGS:
        config_name = config["name"]
        logger.info("\n" + "=" * 60)
        logger.info("測試配置: %s", config_name)
        logger.info("說明: %s", config["description"])
        logger.info("=" * 60)

        # 建立後端
        UltraBackend = BackendRegistry.get("ultra")
        if "preset" in config:
            color_filter = config.get("color_filter")
            backend = UltraBackend.from_preset(
                config["preset"],
                color_filter=color_filter,
            )
        else:
            custom = config["custom"]
            backend = UltraBackend(
                strength=custom["strength"],
                color_filter=custom.get("color_filter"),
                use_trimap_refine=custom.get("use_trimap_refine", True),
            )

        # 建立輸出目錄
        config_output_dir = output_dir / config_name
        config_output_dir.mkdir(parents=True, exist_ok=True)

        # 處理每張圖片
        config_results: dict[str, Any] = {
            "description": config["description"],
            "images": {},
        }

        for image_path in image_files:
            output_path = config_output_dir / image_path.name
            logger.info("處理: %s", image_path.name)

            # 執行處理
            success = backend.process(image_path, output_path)

            if success and output_path.exists():
                # 分析品質
                quality = analyze_image_quality(image_path, output_path)
                config_results["images"][image_path.name] = quality
                logger.info(
                    "  總分: %.1f | 綠幕殘留: %.1f | 邊緣: %.1f | 過度移除: %.1f",
                    quality["total_score"],
                    quality["green_spill"]["score"],
                    quality["edge_bleeding"]["score"],
                    quality["over_removal"]["score"],
                )
            else:
                config_results["images"][image_path.name] = {"error": "處理失敗"}
                logger.error("  處理失敗!")

        # 計算配置平均分數
        scores = [
            img["total_score"]
            for img in config_results["images"].values()
            if "total_score" in img
        ]
        config_results["average_score"] = sum(scores) / len(scores) if scores else 0

        results["configs"][config_name] = config_results
        logger.info("\n配置 %s 平均分數: %.1f", config_name, config_results["average_score"])

    return results


def print_summary(results: dict[str, Any]) -> None:
    """列印測試摘要"""
    print("\n" + "=" * 80)
    print("📊 測試結果摘要".center(80))
    print("=" * 80)

    # 排序配置（按平均分數）
    sorted_configs = sorted(
        results["configs"].items(),
        key=lambda x: x[1].get("average_score", 0),
        reverse=True,
    )

    print("\n排名 | 配置名稱 | 平均分數 | 說明")
    print("-" * 80)

    for i, (name, config) in enumerate(sorted_configs, 1):
        avg = config.get("average_score", 0)
        desc = config.get("description", "")
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "  # noqa: PLR2004
        print(f"{medal} {i}. | {name:25s} | {avg:6.1f} | {desc}")

    print("\n" + "=" * 80)

    # 最佳配置詳細資訊
    if sorted_configs:
        best_name, best_config = sorted_configs[0]
        print(f"\n🏆 最佳配置: {best_name}")
        print(f"   說明: {best_config['description']}")
        print(f"   平均分數: {best_config['average_score']:.1f}")

        # 各圖片得分
        print("\n   各圖片得分:")
        for img_name, img_result in best_config["images"].items():
            if "total_score" in img_result:
                print(f"     {img_name}: {img_result['total_score']:.1f}")

    print("\n" + "=" * 80)


def main() -> None:
    """主程式"""
    # 路徑設定
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "test_file"
    output_dir = project_root / "test_output"

    if not input_dir.exists():
        logger.error("測試目錄不存在: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 執行測試
    results = run_test(input_dir, output_dir)

    # 儲存完整報告
    report_path = output_dir / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("完整報告已儲存: %s", report_path)

    # 列印摘要
    print_summary(results)


if __name__ == "__main__":
    main()
