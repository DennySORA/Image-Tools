"""
產生合成測試圖片

用於測試背景移除功能的各種場景
"""

import numpy as np
from PIL import Image, ImageDraw


def create_solid_background_image(
    size: tuple[int, int] = (1024, 1024), bg_color: tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    建立純色背景圖片，中間有一個圓形前景

    Args:
        size: 圖片尺寸 (width, height)
        bg_color: 背景顏色 (R, G, B)

    Returns:
        RGB 圖片 (H, W, 3), uint8
    """
    width, height = size
    image = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(image)

    # 畫一個中央圓形前景（深藍色）
    center = (width // 2, height // 2)
    radius = min(width, height) // 3

    # 主圓形
    bbox = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
    draw.ellipse(bbox, fill=(50, 100, 200))

    # 添加漸變效果（模擬半透明邊緣）
    for i in range(10):
        alpha = 1.0 - (i / 10.0)
        color = tuple(int(50 * alpha + bg_color[j] * (1 - alpha)) for j in range(3))
        r = radius + i * 3
        bbox_grad = (
            center[0] - r,
            center[1] - r,
            center[0] + r,
            center[1] + r,
        )
        draw.ellipse(bbox_grad, outline=color, width=2)

    return np.array(image)


def create_portrait_like_image(size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    建立類似人像的圖片（帶有頭髮狀邊緣）

    Args:
        size: 圖片尺寸 (width, height)

    Returns:
        RGB 圖片 (H, W, 3), uint8
    """
    width, height = size
    image = Image.new("RGB", size, (230, 230, 250))  # 淺紫背景
    draw = ImageDraw.Draw(image)

    # 建立一個橢圓形「頭部」
    center = (width // 2, height // 2)
    head_width = width // 4
    head_height = int(height * 0.35)

    head_bbox = (
        center[0] - head_width,
        center[1] - head_height,
        center[0] + head_width,
        center[1] + head_height,
    )
    draw.ellipse(head_bbox, fill=(220, 180, 150))  # 膚色

    # 添加「頭髮」（細密的線條模擬）
    rng = np.random.RandomState(42)

    # 頂部頭髮
    for _ in range(100):
        start_x = center[0] + rng.randint(-head_width, head_width)
        start_y = center[1] - head_height + rng.randint(-20, 10)

        end_x = start_x + rng.randint(-30, 30)
        end_y = start_y - rng.randint(30, 80)

        draw.line([(start_x, start_y), (end_x, end_y)], fill=(40, 30, 20), width=1)

    # 側邊頭髮
    for side in [-1, 1]:
        for _ in range(50):
            start_x = center[0] + side * head_width + rng.randint(-10, 10)
            start_y = center[1] + rng.randint(-head_height // 2, head_height // 2)

            end_x = start_x + side * rng.randint(20, 50)
            end_y = start_y + rng.randint(-30, 30)

            draw.line([(start_x, start_y), (end_x, end_y)], fill=(40, 30, 20), width=1)

    return np.array(image)


def create_multi_color_background(size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    建立多色背景圖片

    Args:
        size: 圖片尺寸 (width, height)

    Returns:
        RGB 圖片 (H, W, 3), uint8
    """
    width, height = size

    # 建立漸變背景（三色）使用 numpy
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    # 創建三色漸變
    r = (1 - xx) * 200  # 左側紅色
    g = np.where((xx > 0.33) & (xx < 0.66), 200, 0)  # 中間綠色  # noqa: PLR2004
    b = xx * 200  # 右側藍色

    image = np.stack([r, g, b], axis=-1).astype(np.uint8)

    # 轉換為 PIL 做後續處理
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # 添加中央前景（黃色圓形）
    center = (width // 2, height // 2)
    radius = min(width, height) // 4
    bbox = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
    draw.ellipse(bbox, fill=(200, 200, 50))

    return np.array(pil_image)


def create_complex_edges_image(size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    建立複雜邊緣圖片（測試 trimap）

    Args:
        size: 圖片尺寸 (width, height)

    Returns:
        RGB 圖片 (H, W, 3), uint8
    """
    width, height = size
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 畫一個複雜的星形
    center = (width // 2, height // 2)
    radius = min(width, height) // 3

    points = []
    for i in range(10):
        angle = i * np.pi / 5
        r = radius if i % 2 == 0 else radius * 0.5
        x = center[0] + int(r * np.cos(angle))
        y = center[1] + int(r * np.sin(angle))
        points.append((x, y))

    draw.polygon(points, fill=(100, 150, 200))

    return np.array(image)


def generate_all_test_images() -> dict[str, np.ndarray]:
    """
    產生所有測試圖片

    Returns:
        圖片字典 {名稱: 圖片}
    """
    images = {}

    print("Generating test images...")

    # 1. 純色背景（不同顏色）
    print("  - Solid backgrounds (green, white, black)...")
    images["green_background_1024"] = create_solid_background_image(
        (1024, 1024), (0, 255, 0)
    )
    images["white_background_1024"] = create_solid_background_image(
        (1024, 1024), (255, 255, 255)
    )
    images["black_background_1024"] = create_solid_background_image(
        (1024, 1024), (0, 0, 0)
    )

    # 2. 不同解析度
    print("  - Different resolutions (512, 2048)...")
    images["green_background_512"] = create_solid_background_image(
        (512, 512), (0, 255, 0)
    )
    images["green_background_2048"] = create_solid_background_image(
        (2048, 2048), (0, 255, 0)
    )

    # 3. 人像風格
    print("  - Portrait-like images...")
    images["portrait_like_1024"] = create_portrait_like_image((1024, 1024))
    images["portrait_like_512"] = create_portrait_like_image((512, 512))

    # 4. 多色背景
    print("  - Multi-color background...")
    images["multi_color_1024"] = create_multi_color_background((1024, 1024))

    # 5. 複雜邊緣
    print("  - Complex edges...")
    images["complex_edges_1024"] = create_complex_edges_image((1024, 1024))

    return images


def save_test_images(output_dir: str = "tests/fixtures/synthetic") -> None:
    """
    儲存所有測試圖片

    Args:
        output_dir: 輸出目錄
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    images = generate_all_test_images()

    print(f"\nSaving images to {output_dir}/...")
    for name, image in images.items():
        output_path = os.path.join(output_dir, f"{name}.png")
        # 轉換 numpy array 到 PIL Image 並儲存
        pil_image = Image.fromarray(image)
        pil_image.save(output_path, "PNG")
        print(f"  ✓ {name}.png")

    print(f"\n✅ Generated {len(images)} test images successfully!")


if __name__ == "__main__":
    save_test_images()
