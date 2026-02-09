"""
圖片分割工具核心模組

使用 Alpha 通道連通分量檢測來分割透明圖片中的多個物件

已重構為三個模組：
- detector.py: 物件偵測邏輯
- optimizer.py: 閾值優化
- splitter.py: 主要分割流程（本檔案）
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .detector import DetectedObject, detect_components_from_alpha, merge_objects
from .optimizer import (
    AUTO_ALPHA_THRESHOLDS,
    score_candidate,
    should_search_threshold,
    single_object_looks_merged,
)


logger = logging.getLogger(__name__)

# 常數定義
ALPHA_MAX = 255


class SplitImgError(Exception):
    """圖片分割錯誤"""


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """
    分割設定

    Attributes:
        alpha_threshold: Alpha 閾值 (0-255)
        min_area_ratio: 最小面積比例 (相對於圖片大小)
        min_area_px: 最小像素面積
        merge_pad_px: 合併填充像素 (用於合併相近物件)
        crop_padding_px: 裁切填充像素 (保留物件周圍的透明邊距)
        canvas_mode: 畫布模式 (max/tight/original/fixed)
        canvas_size: 固定畫布大小 (僅當 canvas_mode=fixed 時使用)
        smart_threshold: 是否使用智能閾值檢測
    """

    alpha_threshold: int = 1
    min_area_ratio: float = 0.0005
    min_area_px: int = 64
    merge_pad_px: int = 0
    crop_padding_px: int = 8
    canvas_mode: str = "max"
    canvas_size: tuple[int, int] | None = None
    smart_threshold: bool = True


@dataclass(frozen=True, slots=True)
class SplitResult:
    """
    分割結果

    Attributes:
        sprites: 分割後的精靈圖列表
        alpha_threshold: 使用的 Alpha 閾值
        object_count: 檢測到的物件數量
    """

    sprites: tuple[Image.Image, ...]
    alpha_threshold: int
    object_count: int


class ImageSplitter:
    """
    圖片分割器

    使用 Alpha 通道連通分量分析來分割透明圖片中的多個物件
    """

    def __init__(self, config: SplitConfig | None = None) -> None:
        """
        初始化圖片分割器

        Args:
            config: 分割設定，若為 None 則使用預設設定
        """
        self.config = config or SplitConfig()

    def split_image(self, image: Image.Image) -> SplitResult:
        """
        分割圖片

        Args:
            image: 輸入圖片 (需為 RGBA 模式)

        Returns:
            分割結果

        Raises:
            SplitImgError: 分割失敗時
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # 檢測物件
        detected_objects: list[DetectedObject]
        threshold_value: int

        if self.config.smart_threshold:
            detection = self._detect_objects_smart(image)
            detected_objects = detection["objects"]  # type: ignore[assignment]
            threshold_value = detection["alpha_threshold"]  # type: ignore[assignment]
        else:
            objects_result = self._detect_objects(image, self.config.alpha_threshold)
            detected_objects = [] if isinstance(objects_result, int) else objects_result
            threshold_value = self.config.alpha_threshold

        logger.info(
            f"檢測到 {len(detected_objects)} 個物件 (Alpha 閾值: {threshold_value})"
        )

        # 分割並居中
        sprites = self._split_and_center(image, detected_objects)

        return SplitResult(
            sprites=tuple(sprites),
            alpha_threshold=threshold_value,
            object_count=len(detected_objects),
        )

    def process_file(
        self, input_path: Path, output_dir: Path, prefix: str | None = None
    ) -> int:
        """
        處理單個檔案

        Args:
            input_path: 輸入圖片路徑
            output_dir: 輸出目錄
            prefix: 輸出檔名前綴，若為 None 則使用輸入檔名

        Returns:
            分割出的精靈數量

        Raises:
            SplitImgError: 處理失敗時
        """
        try:
            image = Image.open(input_path).convert("RGBA")
        except Exception as e:
            raise SplitImgError(f"無法開啟圖片: {input_path}") from e

        result = self.split_image(image)

        # 建立輸出目錄
        output_dir.mkdir(parents=True, exist_ok=True)

        # 儲存分割後的圖片
        file_prefix = prefix or input_path.stem
        for i, sprite in enumerate(result.sprites, start=1):
            output_path = output_dir / f"{file_prefix}_{i:02d}.png"
            sprite.save(output_path)
            logger.info(f"已儲存: {output_path}")

        return result.object_count

    # === 偵測邏輯（委派給 detector 模組） ===

    def _detect_objects(
        self, image: Image.Image, alpha_threshold: int
    ) -> list[DetectedObject]:
        """檢測物件"""
        w, h = image.size
        if w <= 0 or h <= 0:
            raise SplitImgError("無效的圖片尺寸")

        if not (0 <= alpha_threshold <= ALPHA_MAX):
            raise SplitImgError("Alpha 閾值必須在 [0, 255] 範圍內")

        alpha = image.getchannel("A").tobytes()
        objects = detect_components_from_alpha(
            alpha, width=w, height=h, alpha_threshold=alpha_threshold
        )

        # 過濾小物件
        min_area = max(
            self.config.min_area_px, int((w * h) * self.config.min_area_ratio)
        )
        objects = [obj for obj in objects if obj.area_px >= min_area]

        if not objects:
            raise SplitImgError("未檢測到物件，請嘗試降低閾值")

        # 合併相近物件
        if self.config.merge_pad_px > 0 and len(objects) > 1:
            objects = merge_objects(
                objects,
                merge_pad_px=self.config.merge_pad_px,
                max_width=w,
                max_height=h,
            )

        # 按位置排序 (從上到下，從左到右)
        objects.sort(key=lambda o: (o.bbox.top, o.bbox.left))
        return objects

    # === 智能閾值（委派給 optimizer 模組） ===

    def _detect_objects_smart(
        self, image: Image.Image
    ) -> dict[str, int | list[DetectedObject]]:
        """智能檢測物件 (自動選擇最佳 Alpha 閾值)"""
        base_objects = self._detect_objects(image, self.config.alpha_threshold)

        w, h = image.size
        need_search = should_search_threshold(
            base_objects
        ) or single_object_looks_merged(w, h, base_objects)

        if not need_search:
            return {
                "objects": base_objects,
                "alpha_threshold": self.config.alpha_threshold,
            }

        # 嘗試不同的閾值
        best_threshold = self.config.alpha_threshold
        best_objects = base_objects
        best_score = score_candidate(best_threshold, best_objects, run_len=1)

        for threshold in AUTO_ALPHA_THRESHOLDS:
            try:
                objects = self._detect_objects(image, threshold)
                score = score_candidate(threshold, objects, run_len=1)
                if score > best_score:
                    best_threshold = threshold
                    best_objects = objects
                    best_score = score
            except SplitImgError:
                continue

        return {"objects": best_objects, "alpha_threshold": best_threshold}

    # === 分割與裁切 ===

    def _split_and_center(
        self, image: Image.Image, objects: Iterable[DetectedObject]
    ) -> list[Image.Image]:
        """分割並居中物件"""
        crops = self._crop_objects(image, objects)
        w, h = image.size
        target = self._resolve_canvas_target(crops, original_size=(w, h))
        return self._paste_centered(crops, target_size=target)

    def _crop_objects(
        self, image: Image.Image, objects: Iterable[DetectedObject]
    ) -> list[Image.Image]:
        """裁切物件"""
        crops: list[Image.Image] = []
        pad = self.config.crop_padding_px

        for obj in objects:
            box = obj.bbox
            crop = image.crop((box.left, box.top, box.right, box.bottom))

            if pad <= 0:
                crops.append(crop)
                continue

            # 加上填充
            padded = Image.new(
                "RGBA",
                (crop.size[0] + pad * 2, crop.size[1] + pad * 2),
                (0, 0, 0, 0),
            )
            padded.paste(crop, (pad, pad), crop)
            crops.append(padded)

        if not crops:
            raise SplitImgError("沒有物件可分割")
        return crops

    def _resolve_canvas_target(
        self, crops: list[Image.Image], *, original_size: tuple[int, int]
    ) -> tuple[int, int] | None:
        """解析畫布目標大小"""
        mode = self.config.canvas_mode.lower().strip()

        if mode not in {"max", "tight", "original", "fixed"}:
            raise SplitImgError("canvas_mode 必須是: max, tight, original, fixed")

        if mode == "tight":
            return None
        if mode == "original":
            return original_size
        if mode == "fixed":
            if self.config.canvas_size is None:
                raise SplitImgError("canvas_mode=fixed 時必須指定 canvas_size")
            return self.config.canvas_size

        # mode == "max"
        return (max(im.size[0] for im in crops), max(im.size[1] for im in crops))

    def _paste_centered(
        self, crops: list[Image.Image], *, target_size: tuple[int, int] | None
    ) -> list[Image.Image]:
        """將裁切的物件貼到居中的畫布上"""
        out: list[Image.Image] = []
        mode = self.config.canvas_mode.lower().strip()

        for crop in crops:
            if mode == "tight" or target_size is None:
                cw, ch = crop.size
            else:
                cw, ch = target_size

            canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            x = (cw - crop.size[0]) // 2
            y = (ch - crop.size[1]) // 2
            canvas.paste(crop, (x, y), crop)
            out.append(canvas)

        return out
