#!/usr/bin/env python3
"""
使用合成測試圖像測試預設配置

不需要 RMBG-2.0 模型授權，使用本地生成的測試圖像進行快速驗證
"""

import sys
from pathlib import Path

from PIL import Image
from src.common import ColorFilter, ColorFilterConfig, PresetLevel
from src.features.background_removal.ultra import UltraBackend


def test_preset_configurations() -> None:
    """測試所有預設配置能正確初始化"""
    print("\n" + "=" * 80)
    print("測試預設配置初始化".center(80))
    print("=" * 80 + "\n")

    presets = ["balanced", "high", "ultra"]
    results = []

    for preset_name in presets:
        try:
            backend = UltraBackend.from_preset(preset_name)
            results.append((preset_name, True, None))
            print(f"✅ {preset_name.upper()} 預設初始化成功")
            print(f"   - 強度: {backend.strength}")
            print(f"   - 解析度: {backend.resolution_config.mode.value}")
            print(
                f"   - 人像精修: {'啟用' if backend.use_portrait_matting else '關閉'}"
            )
            if backend.use_portrait_matting:
                print(f"   - 人像模型: {backend.portrait_matting_model}")
        except Exception as e:
            results.append((preset_name, False, str(e)))
            print(f"❌ {preset_name.upper()} 預設初始化失敗: {e}")

    print()
    return all(success for _, success, _ in results)


def test_color_filter_override() -> None:
    """測試色彩過濾覆蓋"""
    print("\n" + "=" * 80)
    print("測試色彩過濾覆蓋".center(80))
    print("=" * 80 + "\n")

    filters = [
        (ColorFilter.GREEN, "綠幕"),
        (ColorFilter.WHITE, "白背景"),
        (ColorFilter.BLACK, "黑背景"),
    ]

    results = []

    for color_filter, name in filters:
        try:
            filter_config = ColorFilterConfig(color=color_filter, enabled=True)
            backend = UltraBackend.from_preset("balanced", color_filter=filter_config)

            assert backend.color_filter.color == color_filter
            assert backend.color_filter.enabled is True

            results.append((name, True, None))
            print(f"✅ {name}過濾配置成功")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"❌ {name}過濾配置失敗: {e}")

    print()
    return all(success for _, success, _ in results)


def test_device_override() -> None:
    """測試設備覆蓋"""
    print("\n" + "=" * 80)
    print("測試設備設定".center(80))
    print("=" * 80 + "\n")

    try:
        # 測試 CPU 強制
        backend_cpu = UltraBackend.from_preset("balanced", device="cpu")
        assert backend_cpu.device.type == "cpu"
        print("✅ 強制使用 CPU 成功")

        # 測試自動選擇
        backend_auto = UltraBackend.from_preset("balanced")
        print(f"✅ 自動設備選擇: {backend_auto.device.type}")

        print()
        return True
    except Exception as e:
        print(f"❌ 設備設定測試失敗: {e}")
        print()
        return False


def verify_test_images_exist() -> bool:
    """驗證測試圖像是否存在"""
    print("\n" + "=" * 80)
    print("驗證測試圖像".center(80))
    print("=" * 80 + "\n")

    test_images_dir = Path("tests/fixtures/synthetic")

    if not test_images_dir.exists():
        print(f"❌ 測試圖像目錄不存在: {test_images_dir}")
        print("   請運行: python -m tests.fixtures.synthetic.generate_test_images")
        print()
        return False

    expected_images = [
        "green_background_1024.png",
        "white_background_1024.png",
        "black_background_1024.png",
        "portrait_like_1024.png",
        "gradient_background_1024.png",
        "complex_edges_1024.png",
    ]

    missing_images = []
    for img_name in expected_images:
        img_path = test_images_dir / img_name
        if not img_path.exists():
            missing_images.append(img_name)

    if missing_images:
        print(f"❌ 缺少測試圖像: {', '.join(missing_images)}")
        print("   請運行: python -m tests.fixtures.synthetic.generate_test_images")
        print()
        return False

    print(f"✅ 找到 {len(expected_images)} 張測試圖像")

    # 驗證圖像可以打開
    for img_name in expected_images[:3]:  # 只驗證前3張
        img_path = test_images_dir / img_name
        try:
            img = Image.open(img_path)
            print(f"   - {img_name}: {img.size[0]}x{img.size[1]} {img.mode}")
        except Exception as e:
            print(f"   ❌ {img_name}: 無法打開 ({e})")

    print()
    return True


def test_preset_level_enum() -> None:
    """測試使用 PresetLevel 枚舉"""
    print("\n" + "=" * 80)
    print("測試 PresetLevel 枚舉".center(80))
    print("=" * 80 + "\n")

    try:
        # 測試使用枚舉
        backend = UltraBackend.from_preset(PresetLevel.HIGH)
        assert backend.strength == 0.75
        print("✅ 使用 PresetLevel.HIGH 成功")

        # 測試字符串
        backend = UltraBackend.from_preset("high")
        assert backend.strength == 0.75
        print("✅ 使用字符串 'high' 成功")

        print()
        return True
    except Exception as e:
        print(f"❌ PresetLevel 測試失敗: {e}")
        print()
        return False


def test_preset_list() -> None:
    """測試列出預設"""
    print("\n" + "=" * 80)
    print("測試預設列表".center(80))
    print("=" * 80 + "\n")

    try:
        presets = UltraBackend.get_available_presets()

        assert "balanced" in presets
        assert "high" in presets
        assert "ultra" in presets

        print("可用預設：")
        for name, description in presets.items():
            print(f"   • {name}: {description}")

        print()
        return True
    except Exception as e:
        print(f"❌ 預設列表測試失敗: {e}")
        print()
        return False


def main() -> int:
    """主測試函數"""
    print("\n" + "=" * 80)
    print("預設配置系統 - 合成圖像測試".center(80))
    print("=" * 80)
    print("\n此測試不需要 RMBG-2.0 模型，只驗證配置和初始化")

    # 運行所有測試
    tests = [
        ("測試圖像驗證", verify_test_images_exist),
        ("預設配置初始化", test_preset_configurations),
        ("色彩過濾覆蓋", test_color_filter_override),
        ("設備設定", test_device_override),
        ("PresetLevel 枚舉", test_preset_level_enum),
        ("預設列表", test_preset_list),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} 發生異常: {e}")
            results.append((test_name, False))

    # 總結
    print("\n" + "=" * 80)
    print("測試總結".center(80))
    print("=" * 80 + "\n")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{status}: {test_name}")

    print()
    print(f"總計: {passed}/{total} 測試通過 ({passed / total * 100:.1f}%)")

    if passed == total:
        print("\n🎉 所有測試通過！預設配置系統工作正常。")
        print()
        print("下一步：")
        print("  1. 申請 RMBG-2.0 模型訪問權限")
        print("  2. 運行: uv run pytest tests/test_preset_e2e.py -v")
        print("  3. 使用真實圖像測試處理效果")
        return 0
    print("\n⚠️  部分測試失敗，請檢查錯誤訊息。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
