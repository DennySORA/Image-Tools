"""測試圖片分割器功能"""

from pathlib import Path

from PIL import Image

from src.utils.splitter import ImageSplitter, SplitConfig


def test_simple_sprites() -> None:
    """測試簡單精靈圖分割"""
    print("\n=== 測試簡單精靈圖分割 ===")

    input_path = Path("test_images/simple_sprites.png")
    output_dir = Path("test_output/simple")

    # 載入圖片
    image = Image.open(input_path).convert("RGBA")
    print(f"✓ 載入圖片: {input_path} ({image.size[0]}x{image.size[1]})")

    # 建立分割器 (使用預設設定)
    config = SplitConfig(
        canvas_mode="max",
        smart_threshold=True,
    )
    splitter = ImageSplitter(config)

    # 分割圖片
    result = splitter.split_image(image)
    print(f"✓ 檢測到 {result.object_count} 個物件 (Alpha 閾值: {result.alpha_threshold})")

    # 儲存結果
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, sprite in enumerate(result.sprites, start=1):
        output_path = output_dir / f"sprite_{i:02d}.png"
        sprite.save(output_path)
        print(f"  儲存: {output_path} ({sprite.size[0]}x{sprite.size[1]})")

    print(f"✓ 完成！輸出至: {output_dir.absolute()}")


def test_complex_sprites() -> None:
    """測試複雜精靈圖分割"""
    print("\n=== 測試複雜精靈圖分割 ===")

    input_path = Path("test_images/complex_sprites.png")
    output_dir = Path("test_output/complex")

    image = Image.open(input_path).convert("RGBA")
    print(f"✓ 載入圖片: {input_path} ({image.size[0]}x{image.size[1]})")

    config = SplitConfig(
        canvas_mode="tight",  # 使用緊湊模式
        smart_threshold=True,
        crop_padding_px=4,  # 減少填充
    )
    splitter = ImageSplitter(config)

    result = splitter.split_image(image)
    print(f"✓ 檢測到 {result.object_count} 個物件 (Alpha 閾值: {result.alpha_threshold})")

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, sprite in enumerate(result.sprites, start=1):
        output_path = output_dir / f"sprite_{i:02d}.png"
        sprite.save(output_path)
        print(f"  儲存: {output_path} ({sprite.size[0]}x{sprite.size[1]})")

    print(f"✓ 完成！輸出至: {output_dir.absolute()}")


def test_merged_sprites() -> None:
    """測試合併精靈圖分割 (智能閾值)"""
    print("\n=== 測試合併精靈圖分割 (智能閾值) ===")

    input_path = Path("test_images/merged_sprites.png")
    output_dir = Path("test_output/merged")

    image = Image.open(input_path).convert("RGBA")
    print(f"✓ 載入圖片: {input_path} ({image.size[0]}x{image.size[1]})")

    # 測試智能閾值能否分離低 Alpha 值連接的物件
    config = SplitConfig(
        canvas_mode="max",
        smart_threshold=True,  # 啟用智能閾值
    )
    splitter = ImageSplitter(config)

    result = splitter.split_image(image)
    print(
        f"✓ 檢測到 {result.object_count} 個物件 (Alpha 閾值: {result.alpha_threshold})"
    )
    print(f"  智能閾值檢測: {'成功分離' if result.object_count >= 2 else '未分離'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, sprite in enumerate(result.sprites, start=1):
        output_path = output_dir / f"sprite_{i:02d}.png"
        sprite.save(output_path)
        print(f"  儲存: {output_path} ({sprite.size[0]}x{sprite.size[1]})")

    print(f"✓ 完成！輸出至: {output_dir.absolute()}")


def test_backend_integration() -> None:
    """測試後端整合"""
    print("\n=== 測試後端整合 ===")

    from src.backends.image_splitter import ImageSplitterBackend

    input_path = Path("test_images/simple_sprites.png")
    output_path = Path("test_output/backend/simple.png")

    backend = ImageSplitterBackend(model="max", strength=0.5)
    backend.load_model()

    success = backend.process(input_path, output_path)
    if success:
        output_dir = output_path.parent / output_path.stem
        files = list(output_dir.glob("*.png"))
        print(f"✓ 後端處理成功！生成 {len(files)} 個檔案")
        for f in files:
            print(f"  - {f}")
    else:
        print("✗ 後端處理失敗")


if __name__ == "__main__":
    print("開始測試圖片分割功能...")

    try:
        test_simple_sprites()
        test_complex_sprites()
        test_merged_sprites()
        test_backend_integration()

        print("\n" + "=" * 50)
        print("✓ 所有測試完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback

        traceback.print_exc()
