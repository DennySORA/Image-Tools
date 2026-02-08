"""
Pytest 配置和共用 fixtures
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """創建測試數據目錄"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_images_dir(test_data_dir: Path) -> Path:
    """創建測試圖片目錄"""
    images_dir = test_data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    return images_dir


@pytest.fixture(scope="session")
def test_output_dir(test_data_dir: Path) -> Path:
    """創建測試輸出目錄"""
    output_dir = test_data_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def simple_portrait_image(test_images_dir: Path) -> Path:
    """
    生成簡單的人像測試圖片（白底圓形人臉）

    模擬：簡單的人臉輪廓，白色背景
    """
    img_path = test_images_dir / "simple_portrait.png"

    # 創建白色背景
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 繪製圓形臉部（模擬人臉）
    draw.ellipse(
        [(156, 128), (356, 328)],  # 臉部圓形
        fill=(255, 220, 177),  # 膚色
        outline=(200, 180, 150),
        width=2,
    )

    # 繪製眼睛
    draw.ellipse([(200, 200), (220, 220)], fill=(50, 50, 50))  # 左眼
    draw.ellipse([(292, 200), (312, 220)], fill=(50, 50, 50))  # 右眼

    # 繪製嘴巴
    draw.arc([(220, 250), (292, 290)], start=0, end=180, fill=(200, 100, 100), width=3)

    # 繪製頭髮（模擬複雜邊緣）
    for i in range(20):
        x = 156 + i * 10
        y = 128 - np.random.randint(10, 30)
        draw.ellipse([(x, y), (x + 15, y + 20)], fill=(50, 30, 20))

    img.save(img_path)
    return img_path


@pytest.fixture(scope="session")
def greenscreen_image(test_images_dir: Path) -> Path:
    """
    生成綠幕測試圖片

    模擬：綠色背景上的物體
    """
    img_path = test_images_dir / "greenscreen.png"

    # 創建綠色背景（色相約 120，飽和度高）
    img = Image.new("RGB", (512, 512), color=(0, 177, 64))  # 標準綠幕色
    draw = ImageDraw.Draw(img)

    # 繪製前景物體（矩形 + 圓形組合）
    draw.rectangle([(150, 200), (350, 400)], fill=(255, 100, 100), outline=(200, 50, 50), width=3)
    draw.ellipse([(200, 150), (300, 250)], fill=(100, 150, 255), outline=(50, 100, 200), width=3)

    # 添加一些綠色溢出效果（模擬真實綠幕問題）
    for i in range(5):
        x = 150 + i * 10
        color = (int(255 * (1 - i * 0.1)), int(100 + 77 * i * 0.1), int(100 + 64 * i * 0.1))
        draw.line([(x, 200), (x, 400)], fill=color, width=2)

    img.save(img_path)
    return img_path


@pytest.fixture(scope="session")
def white_background_image(test_images_dir: Path) -> Path:
    """
    生成純白背景測試圖片（產品攝影風格）

    模擬：產品攝影，純白背景
    """
    img_path = test_images_dir / "white_background.png"

    # 創建純白背景
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 繪製產品（簡單的瓶子形狀）
    # 瓶身
    draw.rectangle([(200, 250), (312, 450)], fill=(100, 149, 237), outline=(70, 130, 180), width=3)
    # 瓶頸
    draw.rectangle([(230, 200), (282, 250)], fill=(100, 149, 237), outline=(70, 130, 180), width=3)
    # 瓶蓋
    draw.rectangle([(220, 180), (292, 200)], fill=(255, 215, 0), outline=(218, 165, 32), width=2)

    # 添加高光
    draw.ellipse([(220, 280), (250, 310)], fill=(200, 220, 255))

    img.save(img_path)
    return img_path


@pytest.fixture(scope="session")
def black_background_image(test_images_dir: Path) -> Path:
    """
    生成純黑背景測試圖片（工作室風格）

    模擬：工作室人像，純黑背景
    """
    img_path = test_images_dir / "black_background.png"

    # 創建純黑背景
    img = Image.new("RGB", (512, 512), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 繪製人形輪廓
    # 頭部
    draw.ellipse([(206, 100), (306, 200)], fill=(255, 220, 177), outline=(200, 180, 150), width=2)
    # 頸部
    draw.rectangle([(236, 200), (276, 250)], fill=(255, 220, 177))
    # 身體
    draw.polygon(
        [(236, 250), (276, 250), (320, 450), (192, 450)],
        fill=(50, 50, 100),
        outline=(30, 30, 80),
    )

    # 添加邊緣高光（模擬打光）
    draw.arc([(206, 100), (306, 200)], start=45, end=135, fill=(255, 255, 255), width=2)

    img.save(img_path)
    return img_path


@pytest.fixture(scope="session")
def complex_edges_image(test_images_dir: Path) -> Path:
    """
    生成複雜邊緣測試圖片（模擬頭髮、毛髮）

    模擬：複雜的毛髮邊緣
    """
    img_path = test_images_dir / "complex_edges.png"

    # 創建漸層背景
    img = Image.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (int(200 + 55 * y / 512), int(220 + 35 * y / 512), int(255))

    draw = ImageDraw.Draw(img)

    # 繪製主體
    draw.ellipse([(156, 156), (356, 356)], fill=(139, 90, 43), outline=(100, 60, 30), width=2)

    # 繪製複雜的毛髮邊緣（用很多細線模擬）
    np.random.seed(42)
    for _ in range(200):
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(20, 60)
        center_x, center_y = 256, 256
        radius = 100

        start_x = int(center_x + radius * np.cos(angle))
        start_y = int(center_y + radius * np.sin(angle))
        end_x = int(start_x + length * np.cos(angle))
        end_y = int(start_y + length * np.sin(angle))

        color = (
            int(139 - np.random.randint(0, 50)),
            int(90 - np.random.randint(0, 30)),
            int(43 - np.random.randint(0, 20)),
        )
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=1)

    img.save(img_path)
    return img_path


@pytest.fixture(scope="session")
def gradient_background_image(test_images_dir: Path) -> Path:
    """
    生成漸層背景測試圖片（較困難場景）

    模擬：漸層背景，測試演算法在複雜背景下的表現
    """
    img_path = test_images_dir / "gradient_background.png"

    # 創建漸層背景
    img = Image.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            # 徑向漸層
            distance = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
            intensity = int(255 * (1 - distance / 362))  # 362 ≈ sqrt(256^2 + 256^2)
            pixels[x, y] = (intensity, int(intensity * 0.8), int(intensity * 1.2))

    draw = ImageDraw.Draw(img)

    # 繪製前景（複雜形狀）
    draw.polygon(
        [(256, 100), (350, 200), (320, 350), (192, 350), (162, 200)],
        fill=(255, 100, 150),
        outline=(200, 50, 100),
    )

    img.save(img_path)
    return img_path


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """為每個測試創建臨時輸出目錄"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(scope="function")
def cleanup_output(temp_output_dir: Path) -> None:
    """測試後清理輸出"""
    yield
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
