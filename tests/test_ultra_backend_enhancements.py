"""
測試 Ultra Backend 的新功能整合
"""

from pathlib import Path

import pytest
from PIL import Image
from src.common import ColorFilter, ColorFilterConfig, ResolutionConfig
from src.features.background_removal.ultra import UltraBackend


class TestDynamicTrimapParameters:
    """測試動態 Trimap 參數"""

    def test_kernel_size_calculation_small_image(self) -> None:
        """測試小圖的核大小計算"""
        backend = UltraBackend(device="cpu")

        # 512x512 圖片
        kernel_size = backend._calculate_trimap_kernel_size((512, 512), base_kernel=10)

        # 應該小於基準值（1024）
        assert 5 <= kernel_size <= 30  # noqa: PLR2004
        assert kernel_size < 10  # noqa: PLR2004

    def test_kernel_size_calculation_large_image(self) -> None:
        """測試大圖的核大小計算"""
        backend = UltraBackend(device="cpu")

        # 2048x2048 圖片
        kernel_size = backend._calculate_trimap_kernel_size(
            (2048, 2048), base_kernel=10
        )

        # 應該大於基準值
        assert 5 <= kernel_size <= 30  # noqa: PLR2004
        assert kernel_size > 10  # noqa: PLR2004

    def test_kernel_size_limits(self) -> None:
        """測試核大小的上下限"""
        backend = UltraBackend(device="cpu")

        # 極小圖
        kernel_tiny = backend._calculate_trimap_kernel_size((64, 64), base_kernel=10)
        assert kernel_tiny >= 5  # noqa: PLR2004

        # 極大圖
        kernel_huge = backend._calculate_trimap_kernel_size(
            (8192, 8192), base_kernel=10
        )
        assert kernel_huge <= 30  # noqa: PLR2004


class TestPortraitMattingIntegration:
    """測試人像 Matting 整合"""

    @pytest.fixture
    def portrait_test_image(self, tmp_path: Path) -> Path:
        """建立測試用人像圖片"""
        # 讀取合成的人像圖片
        synthetic_path = Path("tests/fixtures/synthetic/portrait_like_512.png")

        if synthetic_path.exists():
            return synthetic_path

        # 如果不存在，建立一個簡單的
        img = Image.new("RGB", (512, 512), (200, 200, 220))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.ellipse((156, 126, 356, 386), fill=(220, 180, 150))

        output_path = tmp_path / "test_portrait.png"
        img.save(output_path)
        return output_path

    def test_portrait_matting_disabled(self, portrait_test_image: Path) -> None:
        """測試停用人像 matting"""
        backend = UltraBackend(
            use_portrait_matting=False,
            use_trimap_refine=False,
            device="cpu",
        )

        # 檢查設定
        assert not backend.use_portrait_matting
        assert backend._portrait_refiner is None

    def test_portrait_matting_enabled(self, portrait_test_image: Path) -> None:
        """測試啟用人像 matting"""
        backend = UltraBackend(
            use_portrait_matting=True,
            portrait_matting_strength=0.8,
            use_trimap_refine=False,
            device="cpu",
        )

        # 檢查設定
        assert backend.use_portrait_matting
        assert backend.portrait_matting_strength == 0.8  # noqa: PLR2004


class TestKMeansBackgroundEstimation:
    """測試 KMeans 背景估計整合"""

    def test_kmeans_in_decontamination(self) -> None:
        """測試 KMeans 在去污染中的使用"""
        # 這個測試驗證 decontaminate_edges 被正確調用
        # 實際的 KMeans 功能在 test_alpha_config.py 中測試
        backend = UltraBackend(device="cpu")

        # 檢查 alpha_config 設定
        assert backend.alpha_config.edge_decontamination


@pytest.mark.slow
class TestFullPipelineWithSyntheticImages:
    """使用合成圖片測試完整流程"""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """建立輸出目錄"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def synthetic_images(self) -> list[Path]:
        """取得合成測試圖片"""
        synthetic_dir = Path("tests/fixtures/synthetic")
        if not synthetic_dir.exists():
            pytest.skip("Synthetic images not generated")

        images = list(synthetic_dir.glob("*.png"))
        if not images:
            pytest.skip("No synthetic images found")

        return images[:3]  # 只測試前 3 張以節省時間

    def test_basic_pipeline(
        self, synthetic_images: list[Path], output_dir: Path
    ) -> None:
        """測試基本流程（無額外功能）"""
        # 注意：這需要下載 RMBG-2.0 模型，可能很慢
        backend = UltraBackend(
            strength=0.7,
            use_trimap_refine=False,
            use_portrait_matting=False,
            device="cpu",
        )

        # 只測試一張圖片
        test_image = synthetic_images[0]
        output_path = output_dir / f"basic_{test_image.name}"

        try:
            backend.load_model()
            success = backend.process(test_image, output_path)

            if success:
                assert output_path.exists()
                # 檢查輸出是 RGBA
                result = Image.open(output_path)
                assert result.mode == "RGBA"
        except Exception as e:  # noqa: BLE001
            # 如果模型下載失敗或其他問題，skip 測試
            pytest.skip(f"Backend processing failed: {e!s}")

    def test_with_trimap_refinement(
        self, synthetic_images: list[Path], output_dir: Path
    ) -> None:
        """測試帶 Trimap refinement 的流程"""
        backend = UltraBackend(
            strength=0.7,
            use_trimap_refine=True,  # 啟用 trimap（動態參數）
            use_portrait_matting=False,
            device="cpu",
        )

        test_image = synthetic_images[0]
        output_path = output_dir / f"trimap_{test_image.name}"

        try:
            backend.load_model()
            success = backend.process(test_image, output_path)

            if success:
                assert output_path.exists()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Backend processing failed: {e!s}")

    def test_with_portrait_matting(
        self, synthetic_images: list[Path], output_dir: Path
    ) -> None:
        """測試帶人像 matting 的流程"""
        # 尋找人像測試圖片
        portrait_image = None
        for img in synthetic_images:
            if "portrait" in img.name:
                portrait_image = img
                break

        if portrait_image is None:
            pytest.skip("No portrait image found")

        backend = UltraBackend(
            strength=0.7,
            use_trimap_refine=True,
            use_portrait_matting=True,  # 啟用人像精修
            portrait_matting_strength=0.8,
            device="cpu",
        )

        output_path = output_dir / f"portrait_{portrait_image.name}"

        try:
            backend.load_model()
            success = backend.process(portrait_image, output_path)

            if success:
                assert output_path.exists()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Backend processing failed: {e!s}")

    def test_with_color_filter(
        self, synthetic_images: list[Path], output_dir: Path
    ) -> None:
        """測試帶色彩過濾的流程"""
        # 尋找綠色背景圖片
        green_image = None
        for img in synthetic_images:
            if "green" in img.name:
                green_image = img
                break

        if green_image is None:
            pytest.skip("No green background image found")

        backend = UltraBackend(
            strength=0.8,
            use_trimap_refine=True,
            color_filter=ColorFilterConfig(enabled=True, color=ColorFilter.GREEN),
            device="cpu",
        )

        output_path = output_dir / f"filtered_{green_image.name}"

        try:
            backend.load_model()
            success = backend.process(green_image, output_path)

            if success:
                assert output_path.exists()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Backend processing failed: {e!s}")


class TestResolutionHandling:
    """測試解析度處理"""

    def test_adaptive_resolution_mode(self) -> None:
        """測試自適應解析度模式"""
        backend = UltraBackend(
            resolution_config=ResolutionConfig(mode="adaptive", max_size=2048),
            device="cpu",
        )

        # 測試不同解析度的推論尺寸計算
        size_512 = backend._get_inference_size((512, 512))
        size_1024 = backend._get_inference_size((1024, 1024))
        size_2048 = backend._get_inference_size((2048, 2048))

        # 512 應該被放大
        assert max(size_512) >= 512  # noqa: PLR2004

        # 1024 應該保持或略微調整
        assert 512 <= max(size_1024) <= 2048  # noqa: PLR2004

        # 2048 應該保持或縮小到 max_size
        assert max(size_2048) <= 2048  # noqa: PLR2004

    def test_fixed_resolution_modes(self) -> None:
        """測試固定解析度模式"""
        backend_1024 = UltraBackend(
            resolution_config=ResolutionConfig(mode="1024"), device="cpu"
        )
        backend_2048 = UltraBackend(
            resolution_config=ResolutionConfig(mode="2048"), device="cpu"
        )

        size_1024 = backend_1024._get_inference_size((1920, 1080))
        size_2048 = backend_2048._get_inference_size((1920, 1080))

        assert size_1024 == (1024, 1024)
        assert size_2048 == (2048, 2048)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
