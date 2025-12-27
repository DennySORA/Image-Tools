"""
綠幕後處理模組

專門處理綠幕背景的色度鍵移除和邊緣 despill 處理
支援三層處理：色度鍵 → AI 精細化 → Despill
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


@dataclass
class GreenScreenConfig:
    """
    綠幕處理設定

    Attributes:
        hue_min: 綠色色相最小值 (0-180, OpenCV HSV)
        hue_max: 綠色色相最大值
        saturation_min: 最低飽和度閾值
        value_min: 最低亮度閾值
        despill_strength: Despill 強度 (0.0-1.0)
        edge_blur: 邊緣模糊半徑
        erode_size: 腐蝕核大小 (用於縮小 mask 邊緣)
        feather_amount: 邊緣羽化量
    """

    hue_min: int = 35
    hue_max: int = 85
    saturation_min: int = 40
    value_min: int = 40
    despill_strength: float = 0.7
    edge_blur: int = 3
    erode_size: int = 2
    feather_amount: int = 2


class GreenScreenProcessor:
    """
    綠幕處理器

    實現色度鍵 (Chroma Key) 技術，專門針對綠幕背景優化
    """

    def __init__(self, config: GreenScreenConfig | None = None) -> None:
        """
        初始化綠幕處理器

        Args:
            config: 綠幕處理設定，若為 None 則使用預設值
        """
        self.config = config or GreenScreenConfig()

    def create_green_mask(self, image: np.ndarray) -> np.ndarray:
        """
        建立綠色區域遮罩

        使用 HSV 色彩空間偵測綠色區域

        Args:
            image: BGR 格式的圖片 (numpy array)

        Returns:
            二值化遮罩，綠色區域為 255，其他為 0
        """
        # 轉換到 HSV 色彩空間
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定義綠色範圍
        lower_green = np.array(
            [self.config.hue_min, self.config.saturation_min, self.config.value_min]
        )
        upper_green = np.array([self.config.hue_max, 255, 255])

        # 建立遮罩
        return cv2.inRange(hsv, lower_green, upper_green)

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        優化遮罩邊緣

        使用形態學操作和模糊來平滑遮罩邊緣

        Args:
            mask: 原始二值化遮罩

        Returns:
            優化後的遮罩
        """
        # 形態學閉運算：填補小洞
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # 形態學開運算：移除雜訊
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # 腐蝕：縮小遮罩邊緣，避免綠邊
        if self.config.erode_size > 0:
            kernel_erode = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.config.erode_size, self.config.erode_size)
            )
            mask = cv2.erode(mask, kernel_erode, iterations=1)

        # 邊緣模糊：羽化效果
        if self.config.edge_blur > 0:
            blur_size = self.config.edge_blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask

    def despill_green(self, image: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        """
        移除邊緣的綠色溢出 (Green Spill)

        對於半透明邊緣區域，降低綠色通道來消除綠色污染

        Args:
            image: BGR 格式的圖片
            mask: 綠色區域遮罩 (用於識別邊緣)

        Returns:
            處理後的圖片
        """
        result = image.copy().astype(np.float32)

        b, g, r = result[:, :, 0], result[:, :, 1], result[:, :, 2]

        # 計算紅藍平均值
        rb_avg = (r + b) / 2

        # 找出綠色過高的區域
        green_excess = g - rb_avg

        # 只處理綠色確實過高的像素
        spill_mask = green_excess > 0

        # 計算新的綠色值
        strength = self.config.despill_strength
        new_g = g - (green_excess * strength * spill_mask)

        # 確保值在有效範圍內
        result[:, :, 1] = np.clip(new_g, 0, 255)

        return result.astype(np.uint8)

    def apply_chroma_key(
        self, image: np.ndarray, existing_alpha: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        應用色度鍵處理

        Args:
            image: BGR 格式的圖片
            existing_alpha: 現有的 alpha 通道 (來自 AI 去背)

        Returns:
            (處理後的 BGR 圖片, alpha 通道)
        """
        # 1. 建立綠色遮罩
        green_mask = self.create_green_mask(image)

        # 2. 優化遮罩
        green_mask = self.refine_mask(green_mask)

        # 3. 反轉遮罩得到前景
        alpha = 255 - green_mask

        # 4. 如果有現有的 alpha，合併兩者 (取交集)
        if existing_alpha is not None:
            alpha = np.minimum(alpha, existing_alpha)

        # 5. Despill 處理
        result = self.despill_green(image, green_mask)

        return result, alpha

    def process_image(self, image: Image.Image) -> Image.Image:
        """
        處理 PIL Image

        Args:
            image: PIL Image (RGB 或 RGBA)

        Returns:
            處理後的 RGBA Image
        """
        # 轉換為 numpy array
        if image.mode == "RGBA":
            rgba = np.array(image)
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            existing_alpha = rgba[:, :, 3]
        else:
            rgb = np.array(image.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            existing_alpha = None

        # 應用色度鍵
        result_bgr, alpha = self.apply_chroma_key(bgr, existing_alpha)

        # 轉回 RGB
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # 組合 RGBA
        result_rgba = np.dstack([result_rgb, alpha])

        return Image.fromarray(result_rgba, "RGBA")

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """
        處理圖片檔案

        Args:
            input_path: 輸入檔案路徑
            output_path: 輸出檔案路徑

        Returns:
            處理是否成功
        """
        try:
            image = Image.open(input_path)
            result = self.process_image(image)
            result.save(output_path, "PNG")
        except Exception:
            logger.exception("GreenScreen postprocess failed: %s", input_path.name)
            return False
        else:
            return True
