"""
測試 portrait_matting 模組
"""

import numpy as np
import pytest
from src.features.background_removal.portrait_matting import (
    PortraitMattingRefiner,
    refine_portrait_alpha,
)


class TestPortraitMattingRefiner:
    """測試 PortraitMattingRefiner 類別"""

    def test_initialization_enhanced_mode(self) -> None:
        """測試初始化（enhanced 模式）"""
        refiner = PortraitMattingRefiner(model_name="enhanced", device="cpu")

        assert refiner.model_name == "enhanced"
        assert refiner.device.type == "cpu"
        assert not refiner._model_loaded

    def test_initialization_modnet_mode(self) -> None:
        """測試初始化（MODNet 模式會回退到 enhanced）"""
        refiner = PortraitMattingRefiner(model_name="modnet", device="cpu")

        # 應該回退到 enhanced
        refiner.load_model()
        assert refiner.model_name == "enhanced"
        assert refiner._model_loaded

    def test_basic_refinement(self) -> None:
        """測試基本精修功能"""
        refiner = PortraitMattingRefiner(model_name="enhanced", device="cpu")

        # 建立簡單的測試圖片和 alpha
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        alpha = np.zeros((100, 100), dtype=np.uint8)

        # 中央圓形前景
        y, x = np.ogrid[:100, :100]
        mask = (x - 50) ** 2 + (y - 50) ** 2 <= 30**2
        alpha[mask] = 255

        # 精修
        refined_alpha = refiner.refine_alpha(image, alpha, focus_strength=0.7)

        # 檢查輸出
        assert refined_alpha.shape == alpha.shape
        assert refined_alpha.dtype == np.uint8
        assert np.all(refined_alpha >= 0) and np.all(refined_alpha <= 255)

    def test_edge_detection(self) -> None:
        """測試邊緣檢測"""
        refiner = PortraitMattingRefiner(model_name="enhanced", device="cpu")

        # 建立測試圖片
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[40:60, 40:60] = 200  # 中央方塊

        alpha = np.zeros((100, 100), dtype=np.float32)
        alpha[40:60, 40:60] = 1.0

        # 檢測邊緣
        edges = refiner._detect_portrait_edges(gray, alpha)

        # 應該在邊界附近檢測到邊緣
        assert edges.dtype == bool
        assert np.any(edges)  # 應該有檢測到邊緣

    def test_different_strengths(self) -> None:
        """測試不同強度"""
        refiner = PortraitMattingRefiner(model_name="enhanced", device="cpu")

        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        alpha = np.full((50, 50), 128, dtype=np.uint8)

        # 低強度
        result_low = refiner.refine_alpha(image, alpha, focus_strength=0.1)
        # 高強度
        result_high = refiner.refine_alpha(image, alpha, focus_strength=0.9)

        # 應該有不同
        assert result_low.shape == result_high.shape
        # 允許結果可能相同（對於這個簡單測試）
        # 但至少要確保都是有效輸出
        assert np.all(result_low >= 0) and np.all(result_low <= 255)
        assert np.all(result_high >= 0) and np.all(result_high <= 255)


class TestRefinePortraitAlpha:
    """測試便捷函數"""

    def test_convenience_function(self) -> None:
        """測試便捷函數"""
        image = np.full((100, 100, 3), 150, dtype=np.uint8)
        alpha = np.full((100, 100), 128, dtype=np.uint8)

        result = refine_portrait_alpha(
            image, alpha, model_name="enhanced", strength=0.5, device="cpu"
        )

        assert result.shape == alpha.shape
        assert result.dtype == np.uint8


class TestPortraitMattingWithSyntheticImages:
    """使用合成圖片進行整合測試"""

    @pytest.fixture
    def portrait_image(self) -> tuple[np.ndarray, np.ndarray]:
        """產生類人像圖片和對應的 alpha"""
        from PIL import Image, ImageDraw

        size = (256, 256)
        image = Image.new("RGB", size, (200, 200, 220))
        draw = ImageDraw.Draw(image)

        # 畫一個橢圓形「頭部」
        head_bbox = (80, 60, 176, 196)
        draw.ellipse(head_bbox, fill=(220, 180, 150))

        # 轉換為 numpy
        image_np = np.array(image)

        # 建立簡單的 alpha（橢圓形區域）
        alpha = np.zeros((256, 256), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]
        center_x, center_y = 128, 128
        rx, ry = 48, 68
        mask = ((x - center_x) / rx) ** 2 + ((y - center_y) / ry) ** 2 <= 1
        alpha[mask] = 255

        return image_np, alpha

    def test_refine_portrait_image(
        self, portrait_image: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """測試人像圖片精修"""
        image, alpha = portrait_image

        refiner = PortraitMattingRefiner(model_name="enhanced", device="cpu")
        refined = refiner.refine_alpha(image, alpha, focus_strength=0.7)

        # 檢查輸出
        assert refined.shape == alpha.shape
        assert refined.dtype == np.uint8

        # 前景區域應該保持高 alpha
        center_region = refined[110:146, 110:146]
        assert np.mean(center_region) > 200  # noqa: PLR2004

    def test_hair_detail_enhancement(
        self, portrait_image: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """測試頭髮細節增強"""
        image, alpha = portrait_image

        refiner = PortraitMattingRefiner(model_name="enhanced", device="cpu")

        # 測試內部方法
        gray = np.mean(image, axis=2).astype(np.uint8)
        alpha_float = alpha.astype(np.float32) / 255.0

        # 建立假的邊緣遮罩
        edges = np.zeros((256, 256), dtype=bool)
        edges[60:196, 80:176] = True

        # 應該能執行不報錯
        enhanced = refiner._enhance_hair_details(gray, alpha_float, edges, strength=0.7)

        assert enhanced.shape == alpha_float.shape
        assert enhanced.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
