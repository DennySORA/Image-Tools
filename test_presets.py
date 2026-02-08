#!/usr/bin/env python3
"""
測試預設配置系統

展示如何使用三個簡化的預設等級：
- Balanced（中等）
- High（高）
- Ultra（最強）
"""

import sys
from pathlib import Path


# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent))

from src.common import print_preset_comparison
from src.features.background_removal.ultra import UltraBackend


def test_preset_info() -> None:
    """列印預設配置資訊"""
    print("\n" + "=" * 80)
    print("背景移除預設配置測試".center(80))
    print("=" * 80)

    # 列印對比表
    print_preset_comparison()

    # 列出可用預設
    print("\n可用預設配置：")
    presets = UltraBackend.get_available_presets()
    for name, desc in presets.items():
        print(f"  • {name}: {desc}")


def test_create_backends() -> None:
    """測試建立不同預設的後端"""
    print("\n" + "=" * 80)
    print("測試建立 UltraBackend 實例".center(80))
    print("=" * 80 + "\n")

    levels = ["balanced", "high", "ultra"]

    for level in levels:
        print(f"\n{'=' * 40}")
        print(f"建立 {level.upper()} 預設")
        print("=" * 40)

        try:
            backend = UltraBackend.from_preset(level)
            print(f"✓ 成功建立 {level} 預設")
            print(f"  強度: {backend.strength}")
            print(f"  解析度: {backend.resolution_config.mode.value}")
            print(f"  Trimap 精修: {backend.use_trimap_refine}")
            print(f"  人像精修: {backend.use_portrait_matting}")
            if backend.use_portrait_matting:
                print(f"  人像模型: {backend.portrait_matting_model}")
                print(f"  人像強度: {backend.portrait_matting_strength}")
            print(f"  邊緣去污染: {backend.alpha_config.edge_decontamination}")
            print(f"  去污染強度: {backend.alpha_config.decontamination_strength}")

        except Exception as e:
            print(f"✗ 建立失敗: {e}")


def test_with_color_filter() -> None:
    """測試使用色彩過濾的預設"""
    print("\n" + "=" * 80)
    print("測試帶色彩過濾的預設".center(80))
    print("=" * 80 + "\n")

    from src.common import ColorFilter, ColorFilterConfig

    # 建立綠幕過濾
    green_filter = ColorFilterConfig(color=ColorFilter.GREEN, enabled=True)

    print("建立 HIGH 預設 + 綠幕過濾")
    backend = UltraBackend.from_preset("high", color_filter=green_filter)

    print("✓ 成功建立")
    print(f"  強度: {backend.strength}")
    print(f"  色彩過濾: {backend.color_filter.color.value}")
    print(f"  過濾啟用: {backend.color_filter.enabled}")


def main() -> None:
    """主函數"""
    print("\n" + "=" * 80)
    print("預設配置系統測試".center(80))
    print("=" * 80)

    try:
        # 測試 1: 列印預設資訊
        test_preset_info()

        # 測試 2: 建立不同預設的後端
        test_create_backends()

        # 測試 3: 使用色彩過濾
        test_with_color_filter()

        print("\n" + "=" * 80)
        print("所有測試完成！".center(80))
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
