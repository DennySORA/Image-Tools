"""
測試精靈圖生成器

生成測試用的精靈圖，包含多個透明物件
"""

from pathlib import Path

from PIL import Image, ImageDraw


def create_test_sprite_sheet(output_path: Path) -> None:
    """
    建立測試精靈圖 (包含 4 個圓形物件)

    Args:
        output_path: 輸出路徑
    """
    # 建立 800x600 的透明畫布
    width, height = 800, 600
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 繪製 4 個不同顏色的圓形
    circles = [
        # (x, y, radius, color)
        (150, 150, 80, (255, 100, 100, 255)),  # 紅色
        (550, 150, 90, (100, 255, 100, 255)),  # 綠色
        (150, 400, 70, (100, 100, 255, 255)),  # 藍色
        (550, 400, 85, (255, 255, 100, 255)),  # 黃色
    ]

    for x, y, radius, color in circles:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=None,
        )

    # 儲存圖片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"✓ 已建立測試精靈圖: {output_path}")


def create_complex_sprite_sheet(output_path: Path) -> None:
    """
    建立複雜測試精靈圖 (包含多種形狀)

    Args:
        output_path: 輸出路徑
    """
    # 建立 1000x800 的透明畫布
    width, height = 1000, 800
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 繪製圓形
    draw.ellipse([50, 50, 150, 150], fill=(255, 100, 100, 255))

    # 繪製矩形
    draw.rectangle([250, 50, 400, 150], fill=(100, 255, 100, 255))

    # 繪製多邊形 (三角形)
    draw.polygon([(550, 150), (650, 150), (600, 50)], fill=(100, 100, 255, 255))

    # 繪製橢圓
    draw.ellipse([750, 50, 950, 150], fill=(255, 255, 100, 255))

    # 繪製帶有透明度漸變的圓形
    for i in range(10):
        alpha = int(255 * (i + 1) / 10)
        radius = 40 - i * 2
        center_x, center_y = 100, 400
        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=(255, 150, 150, alpha),
        )

    # 繪製星形
    import math

    def draw_star(
        draw: ImageDraw.ImageDraw,
        center_x: int,
        center_y: int,
        outer_radius: int,
        inner_radius: int,
        points: int,
        color: tuple[int, int, int, int],
    ) -> None:
        """繪製星形"""
        angle = math.pi / points
        coords = []
        for i in range(points * 2):
            r = outer_radius if i % 2 == 0 else inner_radius
            x = center_x + r * math.cos(i * angle - math.pi / 2)
            y = center_y + r * math.sin(i * angle - math.pi / 2)
            coords.append((x, y))
        draw.polygon(coords, fill=color)

    draw_star(draw, 500, 400, 80, 35, 5, (255, 200, 50, 255))

    # 繪製心形
    draw.chord([650, 350, 750, 450], 0, 180, fill=(255, 100, 150, 255))
    draw.chord([750, 350, 850, 450], 0, 180, fill=(255, 100, 150, 255))
    draw.polygon(
        [(700, 425), (800, 425), (750, 500)], fill=(255, 100, 150, 255)
    )

    # 儲存圖片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"✓ 已建立複雜測試精靈圖: {output_path}")


def create_merged_sprites_sheet(output_path: Path) -> None:
    """
    建立合併的精靈圖 (用於測試智能閾值檢測)

    包含低 Alpha 值連接的物件，測試分割器能否正確分離

    Args:
        output_path: 輸出路徑
    """
    width, height = 800, 400
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 繪製兩個圓形
    draw.ellipse([50, 100, 250, 300], fill=(255, 100, 100, 255))
    draw.ellipse([550, 100, 750, 300], fill=(100, 100, 255, 255))

    # 繪製低 Alpha 值的連接線 (模擬陰影或光暈效果)
    for i in range(10):
        alpha = int(30 - i * 2)  # Alpha 從 30 降到 12
        y = 200 - i * 2
        draw.line(
            [(250, y), (550, y)],
            fill=(150, 150, 150, alpha),
            width=4,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"✓ 已建立合併精靈圖: {output_path}")


if __name__ == "__main__":
    # 建立測試資料夾
    test_dir = Path("test_images")

    # 生成測試圖片
    create_test_sprite_sheet(test_dir / "simple_sprites.png")
    create_complex_sprite_sheet(test_dir / "complex_sprites.png")
    create_merged_sprites_sheet(test_dir / "merged_sprites.png")

    print(f"\n✓ 所有測試圖片已生成至: {test_dir.absolute()}")
