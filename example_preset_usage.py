#!/usr/bin/env python3
"""
快速使用範例：預設配置系統

展示如何使用三個簡化的預設等級處理圖片
"""

from src.common import ColorFilter, ColorFilterConfig
from src.features.background_removal.ultra import UltraBackend


def main() -> None:
    """主函數：展示基本用法"""
    print("=" * 80)
    print("背景移除預設配置 - 快速使用範例".center(80))
    print("=" * 80)

    # 範例 1: 使用 Balanced 預設（日常使用）
    print("\n範例 1: Balanced 預設（日常使用）")
    print("-" * 40)
    backend_balanced = UltraBackend.from_preset("balanced")
    print("✓ 建立 Balanced 後端")
    print(f"  強度: {backend_balanced.strength}")
    print(f"  解析度: {backend_balanced.resolution_config.mode.value}")
    print(f"  人像精修: {backend_balanced.use_portrait_matting}")
    print()
    print("使用範例：")
    print("  backend_balanced.load_model()")
    print('  backend_balanced.process("input.png", "output_balanced.png")')

    # 範例 2: 使用 High 預設（專業工作）
    print("\n範例 2: High 預設（專業工作）")
    print("-" * 40)
    backend_high = UltraBackend.from_preset("high")
    print("✓ 建立 High 後端")
    print(f"  強度: {backend_high.strength}")
    print(f"  解析度: {backend_high.resolution_config.mode.value}")
    print(f"  人像精修: {backend_high.use_portrait_matting}")
    print(f"  人像模型: {backend_high.portrait_matting_model}")
    print()
    print("使用範例：")
    print("  backend_high.load_model()")
    print('  backend_high.process("portrait.png", "output_high.png")')

    # 範例 3: 使用 Ultra 預設（極致品質）
    print("\n範例 3: Ultra 預設（極致品質）")
    print("-" * 40)
    backend_ultra = UltraBackend.from_preset("ultra")
    print("✓ 建立 Ultra 後端")
    print(f"  強度: {backend_ultra.strength}")
    print(f"  解析度: {backend_ultra.resolution_config.mode.value}")
    print(f"  人像精修: {backend_ultra.use_portrait_matting}")
    print(f"  人像模型: {backend_ultra.portrait_matting_model} ⭐")
    print()
    print("注意：Ultra 使用 MODNet 專業模型（需額外依賴）")
    print("安裝：uv add onnxruntime huggingface-hub")
    print()
    print("使用範例：")
    print("  backend_ultra.load_model()")
    print('  backend_ultra.process("professional.png", "output_ultra.png")')

    # 範例 4: 使用色彩過濾
    print("\n範例 4: 搭配色彩過濾（綠幕）")
    print("-" * 40)
    green_filter = ColorFilterConfig(color=ColorFilter.GREEN, enabled=True)
    backend_green = UltraBackend.from_preset("high", color_filter=green_filter)
    print("✓ 建立 High 後端 + 綠幕過濾")
    print(f"  強度: {backend_green.strength}")
    print(f"  色彩過濾: {backend_green.color_filter.color.value}")
    print()
    print("使用範例：")
    print("  backend_green.load_model()")
    print('  backend_green.process("greenscreen.png", "output_green.png")')

    # 列印所有可用預設
    print("\n" + "=" * 80)
    print("所有可用預設：")
    print("-" * 40)
    presets = UltraBackend.get_available_presets()
    for name, description in presets.items():
        print(f"  • {name}: {description}")

    print("\n" + "=" * 80)
    print("提示：")
    print("  1. Balanced 適合快速處理大量圖片")
    print("  2. High 適合需要人像精修的專業工作")
    print("  3. Ultra 追求極致品質，處理時間較長")
    print("  4. 可以搭配 color_filter 處理純色背景")
    print("=" * 80)


if __name__ == "__main__":
    main()
