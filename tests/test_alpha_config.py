"""
測試 alpha_config 模組的新功能
"""

import numpy as np
import pytest
from src.common.alpha_config import (
    calculate_adaptive_resolution,
    decontaminate_edges,
    estimate_background_colors_kmeans,
)


class TestEstimateBackgroundColorsKMeans:
    """測試 KMeans 背景色估計"""

    def test_basic_estimation(self) -> None:
        """測試基本背景色估計"""
        # 建立一個簡單的圖片：紅色背景 + 一些噪音
        image = np.full((100, 100, 3), 200.0, dtype=np.float32)
        image[:, :, 0] = 200  # R
        image[:, :, 1] = 50  # G
        image[:, :, 2] = 50  # B

        # 添加一些噪音
        rng = np.random.RandomState(42)
        noise = rng.uniform(-10, 10, (100, 100, 3)).astype(np.float32)
        image = np.clip(image + noise, 0, 255)

        # 背景區域（整張圖）
        background_region = np.ones((100, 100), dtype=bool)

        # 估計背景色
        bg_color = estimate_background_colors_kmeans(image, background_region)

        # 應該接近紅色
        assert bg_color.shape == (1, 3)
        assert bg_color[0, 0] > 180  # R 應該高  # noqa: PLR2004
        assert bg_color[0, 1] < 70  # G 應該低  # noqa: PLR2004
        assert bg_color[0, 2] < 70  # B 應該低  # noqa: PLR2004

    def test_multi_color_background(self) -> None:
        """測試多色背景"""
        image = np.zeros((100, 100, 3), dtype=np.float32)

        # 左半部紅色
        image[:, :50, 0] = 200
        # 右半部藍色
        image[:, 50:, 2] = 200

        background_region = np.ones((100, 100), dtype=bool)

        # 應該能識別出其中一個主要顏色
        bg_color = estimate_background_colors_kmeans(image, background_region)

        assert bg_color.shape == (1, 3)
        # 應該是紅色或藍色的其中之一
        is_red = bg_color[0, 0] > 150 and bg_color[0, 2] < 50  # noqa: PLR2004
        is_blue = bg_color[0, 2] > 150 and bg_color[0, 0] < 50  # noqa: PLR2004
        assert is_red or is_blue

    def test_insufficient_samples_fallback(self) -> None:
        """測試樣本不足時的回退機制"""
        # 只有 10 個像素（少於 min_samples=50）
        image = np.full((10, 1, 3), 150.0, dtype=np.float32)
        background_region = np.ones((10, 1), dtype=bool)

        bg_color = estimate_background_colors_kmeans(
            image, background_region, min_samples=50
        )

        # 應該回退到中位數
        assert bg_color.shape == (1, 3)
        np.testing.assert_array_almost_equal(bg_color[0], [150, 150, 150])


class TestDecontaminateEdgesWithKMeans:
    """測試帶 KMeans 的邊緣去污染"""

    def test_with_kmeans_enabled(self) -> None:
        """測試啟用 KMeans"""
        # 建立測試圖片：白色背景 + 藍色前景
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        alpha = np.zeros((100, 100), dtype=np.uint8)

        # 中央藍色圓形
        y, x = np.ogrid[:100, :100]
        mask = (x - 50) ** 2 + (y - 50) ** 2 <= 20**2
        image[mask] = [50, 100, 200]
        alpha[mask] = 255

        # 模擬半透明邊緣
        edge_mask = ((x - 50) ** 2 + (y - 50) ** 2 > 20**2) & (
            (x - 50) ** 2 + (y - 50) ** 2 <= 23**2
        )
        alpha[edge_mask] = 128

        # 應用去污染（with KMeans）
        result = decontaminate_edges(image, alpha, strength=0.7, use_kmeans=True)

        # 結果應該是有效的圖片
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_with_kmeans_disabled(self) -> None:
        """測試停用 KMeans（使用中位數）"""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        alpha = np.zeros((100, 100), dtype=np.uint8)

        y, x = np.ogrid[:100, :100]
        mask = (x - 50) ** 2 + (y - 50) ** 2 <= 20**2
        image[mask] = [50, 100, 200]
        alpha[mask] = 255

        edge_mask = ((x - 50) ** 2 + (y - 50) ** 2 > 20**2) & (
            (x - 50) ** 2 + (y - 50) ** 2 <= 23**2
        )
        alpha[edge_mask] = 128

        # 應用去污染（without KMeans）
        result = decontaminate_edges(image, alpha, strength=0.7, use_kmeans=False)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_zero_strength(self) -> None:
        """測試強度為 0 時不做處理"""
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        alpha = np.full((50, 50), 128, dtype=np.uint8)

        result = decontaminate_edges(image, alpha, strength=0.0, use_kmeans=True)

        # 應該完全沒變
        np.testing.assert_array_equal(result, image)


class TestCalculateAdaptiveResolution:
    """測試自適應解析度計算"""

    def test_small_image_upscale(self) -> None:
        """測試小圖放大"""
        # 300x300 應該被放大到至少 512
        result = calculate_adaptive_resolution((300, 300))

        max_dim = max(result)
        assert max_dim >= 512  # noqa: PLR2004
        # 應該保持寬高比
        assert result[0] == result[1]

    def test_medium_image_keep(self) -> None:
        """測試中等尺寸圖片保持不變"""
        # 1024x768 應該保持不變
        result = calculate_adaptive_resolution((1024, 768))

        assert result == (1024, 768)

    def test_large_image_downscale(self) -> None:
        """測試大圖縮小"""
        # 4096x4096 應該被縮小到 max_size（預設 2048）
        result = calculate_adaptive_resolution((4096, 4096), max_size=2048)

        max_dim = max(result)
        assert max_dim <= 2048  # noqa: PLR2004
        # 應該保持寬高比
        assert result[0] == result[1]

    def test_custom_max_size(self) -> None:
        """測試自訂最大尺寸"""
        result = calculate_adaptive_resolution((3000, 3000), max_size=1500)

        max_dim = max(result)
        assert max_dim <= 1500  # noqa: PLR2004

    def test_aspect_ratio_preservation(self) -> None:
        """測試寬高比保持"""
        # 16:9 寬高比
        result = calculate_adaptive_resolution((3840, 2160), max_size=1920)

        # 檢查寬高比是否保持（允許一點誤差）
        original_ratio = 3840 / 2160
        result_ratio = result[0] / result[1]
        assert abs(original_ratio - result_ratio) < 0.01  # noqa: PLR2004


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
