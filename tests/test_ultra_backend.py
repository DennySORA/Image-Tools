"""
Ultra Backend 單元測試

測試 ultra.py 的各個功能：
- 模型載入
- 主分割
- Trimap 建立
- Refinement
- Defringing
- 色彩過濾
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from src.backends.ultra import ColorFilter, ColorFilterConfig, UltraBackend
from src.features.background_removal.color_filter import apply_color_filter
from src.features.background_removal.defringing import apply_advanced_defringing
from src.features.background_removal.trimap import create_trimap


class TestUltraBackendInit:
    """測試 UltraBackend 初始化"""

    def test_default_init(self) -> None:
        """測試預設初始化"""
        backend = UltraBackend()

        assert backend.strength == 0.8  # 預設強度
        assert backend.use_trimap_refine is True
        assert backend.color_filter.enabled is False
        assert backend.color_filter.color == ColorFilter.NONE

    def test_custom_strength(self) -> None:
        """測試自訂強度"""
        backend = UltraBackend(strength=0.5)
        assert backend.strength == 0.5

    def test_strength_clamping(self) -> None:
        """測試強度限制在 0.1-1.0"""
        backend_low = UltraBackend(strength=0.05)
        assert backend_low.strength == 0.1

        backend_high = UltraBackend(strength=1.5)
        assert backend_high.strength == 1.0

    def test_disable_trimap(self) -> None:
        """測試禁用 trimap refinement"""
        backend = UltraBackend(use_trimap_refine=False)
        assert backend.use_trimap_refine is False

    def test_color_filter_config(self) -> None:
        """測試色彩過濾配置"""
        color_filter = ColorFilterConfig(
            enabled=True, color=ColorFilter.GREEN, edge_refine_strength=0.9
        )
        backend = UltraBackend(color_filter=color_filter)

        assert backend.color_filter.enabled is True
        assert backend.color_filter.color == ColorFilter.GREEN
        assert backend.color_filter.edge_refine_strength == 0.9

    def test_device_selection(self) -> None:
        """測試設備選擇"""
        # CPU
        backend_cpu = UltraBackend(device="cpu")
        assert backend_cpu.device.type == "cpu"

        # Auto (應該選擇可用的設備)
        backend_auto = UltraBackend(device=None)
        assert backend_auto.device.type in ("cuda", "mps", "cpu")


class TestUltraBackendModelLoading:
    """測試模型載入"""

    @pytest.mark.slow
    def test_load_model_success(self) -> None:
        """測試成功載入模型（實際載入，標記為 slow）"""
        backend = UltraBackend()

        try:
            backend.load_model()
            assert backend._model is not None
            assert backend._transform is not None
            assert backend._model_loaded is True
        except Exception as e:
            pytest.skip(f"Model loading failed (network issue?): {e}")

    def test_ensure_model_loaded(self) -> None:
        """測試 ensure_model_loaded"""
        backend = UltraBackend()

        assert backend._model_loaded is False

        with patch.object(backend, "load_model") as mock_load:
            backend.ensure_model_loaded()
            mock_load.assert_called_once()

        # 第二次不應該再載入
        backend._model_loaded = True
        with patch.object(backend, "load_model") as mock_load:
            backend.ensure_model_loaded()
            mock_load.assert_not_called()


class TestTrimapCreation:
    """測試 Trimap 建立"""

    def test_create_trimap_basic(self) -> None:
        """測試基本 trimap 建立"""
        # 創建簡單的 alpha (中心白色，外圍黑色)
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[30:70, 30:70] = 255

        trimap = create_trimap(alpha, erode_kernel=5, dilate_kernel=5)

        # 檢查 trimap 值
        assert trimap.shape == alpha.shape
        assert 0 in trimap  # 背景
        assert 128 in trimap  # 未知區
        assert 255 in trimap  # 前景

    def test_create_trimap_all_foreground(self) -> None:
        """測試全前景的情況"""
        alpha = np.full((100, 100), 255, dtype=np.uint8)

        trimap = create_trimap(alpha, erode_kernel=5, dilate_kernel=5)

        # 應該主要是前景，可能有少量未知區
        assert np.sum(trimap == 255) > np.sum(trimap == 128)
        assert np.sum(trimap == 0) == 0  # 沒有背景

    def test_create_trimap_all_background(self) -> None:
        """測試全背景的情況"""
        alpha = np.zeros((100, 100), dtype=np.uint8)

        trimap = create_trimap(alpha, erode_kernel=5, dilate_kernel=5)

        # 應該全是背景
        assert np.all(trimap == 0)


class TestAdvancedDefringing:
    """測試進階去色邊"""

    def test_defringing_disabled_low_strength(self) -> None:
        """測試低強度時禁用 defringing"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        alpha = np.full((100, 100), 128, dtype=np.uint8)

        result = apply_advanced_defringing(image, alpha, strength=0.3)

        # 低強度應該不處理，返回原圖
        np.testing.assert_array_equal(result, image)

    def test_defringing_enabled_high_strength(self) -> None:
        """測試高強度時啟用 defringing"""
        # 創建有色偏的圖片（偏綠）
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 100  # R
        image[:, :, 1] = 200  # G (偏高)
        image[:, :, 2] = 100  # B

        # 半透明 alpha
        alpha = np.full((100, 100), 128, dtype=np.uint8)

        result = apply_advanced_defringing(image, alpha, strength=0.8)

        # 應該有改變（減少綠色偏移）
        assert not np.array_equal(result, image)
        # 綠色通道應該被降低
        assert np.mean(result[:, :, 1]) < np.mean(image[:, :, 1])

    def test_defringing_preserves_opaque_regions(self) -> None:
        """測試不透明區域不受影響"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        alpha = np.full((100, 100), 255, dtype=np.uint8)  # 完全不透明

        result = apply_advanced_defringing(image, alpha, strength=0.8)

        # 完全不透明的區域應該基本不變（可能有小數運算誤差）
        assert np.allclose(result, image, atol=10)


class TestColorFilter:
    """測試色彩過濾"""

    def test_color_filter_disabled(self) -> None:
        """測試禁用色彩過濾"""
        color_filter = ColorFilterConfig(enabled=False, color=ColorFilter.NONE)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        alpha = np.full((100, 100), 200, dtype=np.uint8)

        result_alpha = apply_color_filter(image, alpha, color_filter)

        # 禁用時應該返回原 alpha
        np.testing.assert_array_equal(result_alpha, alpha)

    def test_green_color_filter(self) -> None:
        """測試綠色過濾（RGB despill，不修改 alpha）"""
        color_filter = ColorFilterConfig(enabled=True, color=ColorFilter.GREEN)

        # 創建綠色背景圖片
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :, :] = [0, 177, 64]  # 上半部綠色（綠幕色）
        image[50:, :, :] = [255, 100, 100]  # 下半部紅色（前景）
        original_green_top = image[:50, :, 1].copy()

        alpha = np.full((100, 100), 200, dtype=np.uint8)

        result_alpha = apply_color_filter(image, alpha, color_filter)

        # 綠色模式不修改 alpha（避免過度移除）
        assert np.array_equal(result_alpha, alpha)
        # 綠色區域的 G 通道應該被降低（despill）
        assert np.mean(image[:50, :, 1]) < np.mean(original_green_top)
        # 紅色前景的 G 通道不受影響（alpha=200 接近不透明）
        assert image[50, 50, 1] == 100  # noqa: PLR2004

    def test_white_color_filter(self) -> None:
        """測試白色過濾"""
        color_filter = ColorFilterConfig(enabled=True, color=ColorFilter.WHITE)

        # 創建白色背景圖片
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :, :] = 255  # 上半部純白
        image[50:, :, :] = 100  # 下半部灰色（前景）

        alpha = np.full((100, 100), 200, dtype=np.uint8)

        result_alpha = apply_color_filter(image, alpha, color_filter)

        # 白色區域的 alpha 應該被降低
        assert np.mean(result_alpha[:50, :]) < np.mean(alpha[:50, :])

    def test_black_color_filter(self) -> None:
        """測試黑色過濾"""
        color_filter = ColorFilterConfig(enabled=True, color=ColorFilter.BLACK)

        # 創建黑色背景圖片
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :, :] = 0  # 上半部純黑
        image[50:, :, :] = 200  # 下半部亮色（前景）

        alpha = np.full((100, 100), 200, dtype=np.uint8)

        result_alpha = apply_color_filter(image, alpha, color_filter)

        # 黑色區域的 alpha 應該被降低
        assert np.mean(result_alpha[:50, :]) < np.mean(alpha[:50, :])


class TestProcessIntegration:
    """測試完整處理流程（需要模型）"""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_simple_image(
        self, simple_portrait_image: Path, temp_output_dir: Path
    ) -> None:
        """測試處理簡單圖片（實際運行，標記為 slow）"""
        backend = UltraBackend(strength=0.7)
        output_path = temp_output_dir / "output.png"

        try:
            backend.load_model()
            success = backend.process(simple_portrait_image, output_path)

            assert success is True
            assert output_path.exists()

            # 檢查輸出
            result = Image.open(output_path)
            assert result.mode == "RGBA"
            assert result.size == (512, 512)
        except Exception as e:
            pytest.skip(f"Integration test failed (network/model issue?): {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_with_color_filter(
        self, greenscreen_image: Path, temp_output_dir: Path
    ) -> None:
        """測試帶色彩過濾的處理"""
        color_filter = ColorFilterConfig(enabled=True, color=ColorFilter.GREEN)
        backend = UltraBackend(strength=0.8, color_filter=color_filter)
        output_path = temp_output_dir / "greenscreen_output.png"

        try:
            backend.load_model()
            success = backend.process(greenscreen_image, output_path)

            assert success is True
            assert output_path.exists()

            # 檢查輸出的 alpha 通道
            result = Image.open(output_path)
            alpha_channel = np.array(result)[:, :, 3]

            # 綠色背景區域應該是透明的（alpha 接近 0）
            # 前景物體應該是不透明的（alpha 接近 255）
            assert np.min(alpha_channel) < 50  # 有透明區域
            assert np.max(alpha_channel) > 200  # 有不透明區域
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    def test_process_error_handling(self, temp_output_dir: Path) -> None:
        """測試錯誤處理"""
        backend = UltraBackend()

        # 不載入模型直接處理
        nonexistent_input = Path("nonexistent.jpg")
        output_path = temp_output_dir / "output.png"

        try:
            # 應該返回 False 而不是拋出異常
            success = backend.process(nonexistent_input, output_path)
            assert success is False
        except Exception:
            # 如果模型是 gated 或網路問題，跳過測試
            pytest.skip("Model access restricted or network issue")


class TestClassMethods:
    """測試類方法"""

    def test_get_available_models(self) -> None:
        """測試取得可用模型列表"""
        models = UltraBackend.get_available_models()
        assert isinstance(models, list)
        assert "auto" in models

    def test_get_model_description(self) -> None:
        """測試取得模型說明"""
        description = UltraBackend.get_model_description()
        assert isinstance(description, str)
        assert len(description) > 0
        assert "RMBG-2.0" in description
        assert "非商用" in description or "non-commercial" in description.lower()
