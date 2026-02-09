"""
Gemini 浮水印移除後端

使用反向 Alpha 混合演算法移除 Google Gemini AI 生成圖片中的浮水印。

原理:
    Gemini 添加浮水印: watermarked = α × logo + (1 - α) × original
    反向求解原始像素: original = (watermarked - α × logo) / (1 - α)

參考: https://github.com/journey-ad/gemini-watermark-remover
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from src.backends.registry import BackendRegistry
from src.core.interfaces import BaseBackend


logger = logging.getLogger(__name__)

# 常數定義
ALPHA_THRESHOLD: float = 0.002  # 忽略極小的 alpha 值 (雜訊)
MAX_ALPHA: float = 0.99  # 避免除以接近零的值
LOGO_VALUE: int = 255  # 白色浮水印參考值
MIN_PIXEL_CHANNELS: int = 3  # RGB 最小通道數

# 參考圖片目錄
ASSETS_DIR: Path = Path(__file__).parent / "assets"

# Gemini 浮水印尺寸切換閾值 (寬高皆大於此值時使用大尺寸)
WATERMARK_SIZE_THRESHOLD: int = 1024

# 可用模式
AVAILABLE_MODES: tuple[str, ...] = ("auto", "48px", "96px")
DEFAULT_MODE: str = "auto"


@dataclass(frozen=True)
class WatermarkConfig:
    """浮水印配置。"""

    logo_size: int
    margin_right: int
    margin_bottom: int


def _load_reference_image(size: int) -> Image.Image:
    """
    載入浮水印參考圖片。

    Args:
        size: 浮水印尺寸 (48 或 96)

    Returns:
        參考圖片

    Raises:
        FileNotFoundError: 當參考圖片不存在時
    """
    path = ASSETS_DIR / f"bg_{size}.png"
    if not path.exists():
        raise FileNotFoundError(f"參考圖片不存在: {path}")
    return Image.open(path).convert("RGB")


def _calculate_alpha_map(bg_image: Image.Image) -> np.ndarray:
    """
    從參考背景圖片計算 alpha 通道映射。

    對每個像素取 RGB 三通道最大值並正規化至 [0, 1]。

    Args:
        bg_image: 參考背景圖片

    Returns:
        alpha 映射 (NumPy array, float64, shape: H*W)
    """
    pixels = np.array(bg_image)[:, :, :3]  # (H, W, 3)
    result: np.ndarray = pixels.max(axis=2).ravel().astype(np.float64) / 255.0
    return result


def _detect_watermark_config(width: int, height: int) -> WatermarkConfig:
    """
    根據圖片尺寸偵測浮水印配置。

    Gemini 規則: 若寬高皆大於閾值，使用 96×96 浮水印，否則使用 48×48。

    Args:
        width: 圖片寬度
        height: 圖片高度

    Returns:
        浮水印配置
    """
    if width > WATERMARK_SIZE_THRESHOLD and height > WATERMARK_SIZE_THRESHOLD:
        return WatermarkConfig(logo_size=96, margin_right=64, margin_bottom=64)
    return WatermarkConfig(logo_size=48, margin_right=32, margin_bottom=32)


def _calculate_watermark_position(
    image_width: int,
    image_height: int,
    config: WatermarkConfig,
) -> tuple[int, int]:
    """
    計算浮水印在圖片中的位置 (右下角)。

    Args:
        image_width: 圖片寬度
        image_height: 圖片高度
        config: 浮水印配置

    Returns:
        (x, y) 浮水印左上角座標
    """
    x = image_width - config.margin_right - config.logo_size
    y = image_height - config.margin_bottom - config.logo_size
    return (x, y)


@dataclass(frozen=True)
class _WatermarkRemovalParams:
    """浮水印移除參數。"""

    alpha_map: np.ndarray
    x: int
    y: int
    size: int
    strength: float


def _remove_watermark(
    image: Image.Image,
    params: _WatermarkRemovalParams,
) -> Image.Image:
    """
    使用反向 Alpha 混合演算法移除浮水印（NumPy 向量化）。

    對 RGB 三通道進行反向混合，RGBA 圖片同時也還原 alpha 通道。

    Args:
        image: 要處理的圖片
        params: 浮水印移除參數

    Returns:
        處理後的圖片
    """
    has_alpha = image.mode == "RGBA"
    img_arr = np.array(image, dtype=np.float64)

    # 提取浮水印區域
    y1, y2 = params.y, params.y + params.size
    x1, x2 = params.x, params.x + params.size
    region = img_arr[y1:y2, x1:x2].copy()

    # 重塑 alpha_map 為 (size, size)
    wm_alpha = params.alpha_map.reshape(params.size, params.size).copy()

    # 忽略極小的 alpha 值 (雜訊)
    active = wm_alpha >= ALPHA_THRESHOLD
    # 限制 alpha 值以避免除以接近零的值
    wm_alpha = np.clip(wm_alpha, 0.0, MAX_ALPHA)
    one_minus_alpha = 1.0 - wm_alpha

    # 避免除以零
    safe_divisor = np.where(active, one_minus_alpha, 1.0)

    # 對 RGB 三通道進行反向 Alpha 混合
    for c in range(3):
        watermarked = region[:, :, c]
        original = (watermarked - wm_alpha * LOGO_VALUE) / safe_divisor
        blended = watermarked * (1.0 - params.strength) + original * params.strength
        region[:, :, c] = np.where(active, blended, watermarked)

    # RGBA 圖片同時還原 alpha 通道
    if has_alpha and region.shape[2] > MIN_PIXEL_CHANNELS:
        wm_a = region[:, :, 3]
        orig_a = (wm_a - wm_alpha * LOGO_VALUE) / safe_divisor
        blended_a = wm_a * (1.0 - params.strength) + orig_a * params.strength
        region[:, :, 3] = np.where(active, blended_a, wm_a)

    # 限制在 [0, 255] 並寫回
    img_arr[y1:y2, x1:x2] = np.clip(region, 0, 255)

    return Image.fromarray(np.round(img_arr).astype(np.uint8), mode=image.mode)


@BackendRegistry.register("gemini-watermark")
class GeminiWatermarkBackend(BaseBackend):
    """
    Gemini 浮水印移除後端

    使用反向 Alpha 混合演算法移除 Gemini AI 生成圖片中的浮水印。
    支援自訂浮水印位置與大小。
    """

    name: ClassVar[str] = "gemini-watermark"
    description: ClassVar[str] = "Gemini 浮水印移除 - 移除 Gemini AI 生成圖片的浮水印"

    def __init__(self, model: str = DEFAULT_MODE, strength: float = 1.0) -> None:
        """
        初始化 Gemini 浮水印移除後端。

        Args:
            model: 偵測模式 (auto / 48px / 96px)
            strength: 移除強度 (0.1-1.0，預設 1.0 完全移除)

        Raises:
            ValueError: 當模式不支援時
        """
        super().__init__(strength=strength)

        if model not in AVAILABLE_MODES:
            raise ValueError(f"不支援的模式: {model}，可用模式: {AVAILABLE_MODES}")

        self.model = model
        self._alpha_maps: dict[int, np.ndarray] = {}

    def load_model(self) -> None:
        """載入參考圖片並預先計算 alpha 映射。"""
        logger.info("Gemini Watermark mode: %s", self.model)
        logger.info("Gemini Watermark strength: %s", self.strength)

        # 根據模式預載入對應的參考圖片
        sizes_to_load: list[int]
        if self.model == "48px":
            sizes_to_load = [48]
        elif self.model == "96px":
            sizes_to_load = [96]
        else:
            sizes_to_load = [48, 96]

        for size in sizes_to_load:
            bg_image = _load_reference_image(size)
            self._alpha_maps[size] = _calculate_alpha_map(bg_image)
            bg_image.close()

        logger.info("Gemini Watermark reference images loaded")

    def _get_alpha_map(self, logo_size: int) -> np.ndarray:
        """
        取得指定尺寸的 alpha 映射，必要時動態載入。

        Args:
            logo_size: 浮水印尺寸

        Returns:
            alpha 映射 (NumPy array)
        """
        alpha_map = self._alpha_maps.get(logo_size)
        if alpha_map is None:
            bg_image = _load_reference_image(logo_size)
            alpha_map = _calculate_alpha_map(bg_image)
            self._alpha_maps[logo_size] = alpha_map
            bg_image.close()
        return alpha_map

    def process(self, input_path: Path, output_path: Path) -> bool:
        """
        處理單張圖片，移除浮水印。

        Args:
            input_path: 輸入圖片路徑
            output_path: 輸出圖片路徑

        Returns:
            處理是否成功
        """
        self.ensure_model_loaded()

        try:
            image = Image.open(input_path)
            width, height = image.size

            # 決定浮水印配置
            if self.model == "48px":
                wm_config = WatermarkConfig(
                    logo_size=48, margin_right=32, margin_bottom=32
                )
            elif self.model == "96px":
                wm_config = WatermarkConfig(
                    logo_size=96, margin_right=64, margin_bottom=64
                )
            else:
                wm_config = _detect_watermark_config(width, height)

            # 檢查圖片是否足夠大
            min_size = wm_config.logo_size + max(
                wm_config.margin_right, wm_config.margin_bottom
            )
            if width < min_size or height < min_size:
                logger.warning(
                    "圖片太小，無法處理浮水印: %s (%dx%d)",
                    input_path.name,
                    width,
                    height,
                )
                image.save(output_path, "PNG")
                image.close()
                return True

            # 取得 alpha 映射
            alpha_map = self._get_alpha_map(wm_config.logo_size)

            # 計算浮水印位置 (右下角)
            wx, wy = _calculate_watermark_position(width, height, wm_config)

            # 確保圖片為 RGB 或 RGBA 模式
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")  # type: ignore[assignment]

            # 移除浮水印
            result = _remove_watermark(
                image,
                _WatermarkRemovalParams(
                    alpha_map=alpha_map,
                    x=wx,
                    y=wy,
                    size=wm_config.logo_size,
                    strength=self.strength,
                ),
            )

            # 儲存結果
            result.save(output_path, "PNG")
            image.close()
            result.close()

        except Exception:
            logger.exception("Gemini watermark removal failed: %s", input_path.name)
            return False
        else:
            return True

    @classmethod
    def get_available_models(cls) -> list[str]:
        """取得可用模式列表。"""
        return list(AVAILABLE_MODES)

    @classmethod
    def get_model_description(cls) -> str:
        """取得模式說明。"""
        return """
  浮水印偵測模式:
    auto - 自動偵測 (寬高 >1024 使用 96px，否則 48px)
    48px - 強制使用 48×48 浮水印模式
    96px - 強制使用 96×96 浮水印模式
"""
