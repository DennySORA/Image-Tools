"""
圖片分割工具模組

使用 Alpha 通道連通分量檢測來分割透明圖片中的多個物件
"""

import logging
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image

from .geometry import BBox
from .union_find import UnionFind, UnionFindDynamic


logger = logging.getLogger(__name__)


# 常數定義
ALPHA_MAX: Final[int] = 255
AUTO_ALPHA_THRESHOLDS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64)
MIN_OBJECTS_FOR_OUTLIER_CHECK: Final[int] = 2
OUTLIER_RATIO_MERGED_THRESHOLD: Final[float] = 1.6
SINGLE_OBJECT_BBOX_AREA_RATIO_THRESHOLD: Final[float] = 0.30
SINGLE_OBJECT_FILL_RATIO_THRESHOLD: Final[float] = 0.70
MIN_STABLE_RUN_LEN: Final[int] = 2
UNSTABLE_SPLIT_PENALTY: Final[float] = 40.0


class SplitImgError(Exception):
    """圖片分割錯誤"""


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """
    檢測到的物件

    Attributes:
        bbox: 邊界框
        area_px: 像素面積
    """

    bbox: BBox
    area_px: int


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
            # Cast to proper types (dict values can be int | list but keys determine actual type)
            detected_objects = detection["objects"]  # type: ignore[assignment]
            threshold_value = detection["alpha_threshold"]  # type: ignore[assignment]
        else:
            objects_result = self._detect_objects(image, self.config.alpha_threshold)
            # Handle case where _detect_objects returns an error code (int)
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
        objects = self._detect_components_from_alpha(
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
            objects = self._merge_objects(
                objects,
                merge_pad_px=self.config.merge_pad_px,
                max_width=w,
                max_height=h,
            )

        # 按位置排序 (從上到下，從左到右)
        objects.sort(key=lambda o: (o.bbox.top, o.bbox.left))
        return objects

    def _detect_objects_smart(
        self, image: Image.Image
    ) -> dict[str, int | list[DetectedObject]]:
        """智能檢測物件 (自動選擇最佳 Alpha 閾值)"""
        base_objects = self._detect_objects(image, self.config.alpha_threshold)

        # 檢查是否需要搜尋更好的閾值
        should_search = self._should_search_threshold(
            base_objects
        ) or self._single_object_looks_merged(image, base_objects)

        if not should_search:
            return {
                "objects": base_objects,
                "alpha_threshold": self.config.alpha_threshold,
            }

        # 嘗試不同的閾值
        best_threshold = self.config.alpha_threshold
        best_objects = base_objects
        best_score = self._score_candidate(best_threshold, best_objects, run_len=1)

        for threshold in AUTO_ALPHA_THRESHOLDS:
            try:
                objects = self._detect_objects(image, threshold)
                score = self._score_candidate(threshold, objects, run_len=1)
                if score > best_score:
                    best_threshold = threshold
                    best_objects = objects
                    best_score = score
            except SplitImgError:
                continue

        return {"objects": best_objects, "alpha_threshold": best_threshold}

    def _detect_components_from_alpha(
        self, alpha: bytes, *, width: int, height: int, alpha_threshold: int
    ) -> list[DetectedObject]:
        """從 Alpha 通道檢測連通分量"""
        labels, uf, w2 = self._label_alpha_to_buffer(
            alpha, width=width, height=height, alpha_threshold=alpha_threshold
        )
        return self._objects_from_labeled_buffer(
            labels, uf, width=width, height=height, width2=w2
        )

    def _label_alpha_to_buffer(
        self, alpha: bytes, *, width: int, height: int, alpha_threshold: int
    ) -> tuple[array[int], UnionFindDynamic, int]:
        """將 Alpha 通道標記到緩衝區 (8-連通標記)"""
        w: Final[int] = width
        h: Final[int] = height
        w2: Final[int] = w + 2  # 加 1 像素邊框避免邊界檢查
        h2: Final[int] = h + 2

        labels: array[int] = array("I", [0]) * (w2 * h2)
        uf = UnionFindDynamic()

        mv = memoryview(alpha)
        thr: Final[int] = alpha_threshold

        for y in range(h):
            alpha_row = y * w
            label_row = (y + 1) * w2 + 1

            for x in range(w):
                if mv[alpha_row + x] <= thr:
                    continue

                idx = label_row + x

                # 檢查 8 個鄰居 (上、左上、左、右上)
                neighbors = (
                    labels[idx - 1],  # 左
                    labels[idx - w2],  # 上
                    labels[idx - w2 - 1],  # 左上
                    labels[idx - w2 + 1],  # 右上
                )

                base = self._min_nonzero(*neighbors)
                if base == 0:
                    # 新分量
                    labels[idx] = uf.make_set()
                    continue

                labels[idx] = base
                self._union_neighbors(uf, base, neighbors)

        return labels, uf, w2

    def _objects_from_labeled_buffer(
        self,
        labels: array[int],
        uf: UnionFindDynamic,
        *,
        width: int,
        height: int,
        width2: int,
    ) -> list[DetectedObject]:
        """從標記緩衝區提取物件"""
        stats: dict[int, tuple[int, int, int, int, int]] = {}
        # tuple = (left, top, right_excl, bottom_excl, area)

        for y in range(height):
            label_row = (y + 1) * width2 + 1
            for x in range(width):
                idx = label_row + x
                lab = labels[idx]
                if lab == 0:
                    continue

                root = uf.find(lab)
                labels[idx] = root

                entry = stats.get(root)
                if entry is None:
                    stats[root] = (x, y, x + 1, y + 1, 1)
                    continue

                left, top, right, bottom, area = entry
                stats[root] = (
                    min(left, x),
                    min(top, y),
                    max(right, x + 1),
                    max(bottom, y + 1),
                    area + 1,
                )

        return [
            DetectedObject(
                bbox=BBox(left=left, top=top, right=right, bottom=bottom),
                area_px=area,
            )
            for left, top, right, bottom, area in stats.values()
        ]

    def _merge_objects(
        self,
        objects: list[DetectedObject],
        *,
        merge_pad_px: int,
        max_width: int,
        max_height: int,
    ) -> list[DetectedObject]:
        """合併相近的物件"""
        expanded: list[BBox] = [
            obj.bbox.expand(merge_pad_px, max_width=max_width, max_height=max_height)
            for obj in objects
        ]
        uf = UnionFind(len(objects))

        # 檢查重疊
        for i in range(len(objects)):
            bi = expanded[i]
            for j in range(i + 1, len(objects)):
                if bi.overlaps(expanded[j]):
                    uf.union(i, j)

        # 分組
        groups: dict[int, list[int]] = {}
        for i in range(len(objects)):
            root = uf.find(i)
            groups.setdefault(root, []).append(i)

        # 合併
        merged: list[DetectedObject] = []
        for idxs in groups.values():
            bbox = objects[idxs[0]].bbox
            area = 0
            for i in idxs:
                bbox = bbox.union(objects[i].bbox)
                area += objects[i].area_px
            merged.append(DetectedObject(bbox=bbox, area_px=area))

        return merged

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

    # 輔助方法
    @staticmethod
    def _min_nonzero(*values: int) -> int:
        """取最小非零值"""
        base = 0
        for v in values:
            if v and (base == 0 or v < base):
                base = v
        return base

    @staticmethod
    def _union_neighbors(
        uf: UnionFindDynamic, base: int, neighbors: tuple[int, int, int, int]
    ) -> None:
        """合併鄰居"""
        for n in neighbors:
            if n and n != base:
                uf.union(base, n)

    @staticmethod
    def _median_int(values: list[int]) -> int:
        """計算中位數"""
        if not values:
            return 0
        values = sorted(values)
        return values[len(values) // 2]

    def _outlier_ratio(self, objects: list[DetectedObject]) -> float:
        """計算離群值比例"""
        if len(objects) < MIN_OBJECTS_FOR_OUTLIER_CHECK:
            return 1.0

        widths = [o.bbox.width() for o in objects]
        heights = [o.bbox.height() for o in objects]
        bbox_areas = [o.bbox.area() for o in objects]

        mw = max(1, self._median_int(widths))
        mh = max(1, self._median_int(heights))
        ma = max(1, self._median_int(bbox_areas))

        ratios = []
        for w, h, a in zip(widths, heights, bbox_areas, strict=True):
            ratios.append(w / mw)
            ratios.append(h / mh)
            ratios.append(a / ma)

        return max(ratios, default=1.0)

    def _should_search_threshold(self, objects: list[DetectedObject]) -> bool:
        """是否應該搜尋更好的閾值"""
        if len(objects) < MIN_OBJECTS_FOR_OUTLIER_CHECK:
            return False
        return self._outlier_ratio(objects) >= OUTLIER_RATIO_MERGED_THRESHOLD

    def _single_object_looks_merged(
        self, image: Image.Image, objects: list[DetectedObject]
    ) -> bool:
        """單個物件是否看起來像合併的"""
        if len(objects) != 1:
            return False
        w, h = image.size
        image_area = max(1, w * h)
        bbox_area = max(1, objects[0].bbox.area())
        bbox_ratio = bbox_area / image_area
        fill_ratio = objects[0].area_px / bbox_area
        return (
            bbox_ratio >= SINGLE_OBJECT_BBOX_AREA_RATIO_THRESHOLD
            and fill_ratio <= SINGLE_OBJECT_FILL_RATIO_THRESHOLD
        )

    def _score_candidate(
        self, threshold: int, objects: list[DetectedObject], run_len: int
    ) -> float:
        """評分候選結果"""
        n = len(objects)
        if n <= 0:
            return float("-inf")

        outliers = self._outlier_ratio(objects)
        small_cnt = self._small_area_count(objects)
        unstable_penalty = (
            UNSTABLE_SPLIT_PENALTY
            if n >= MIN_OBJECTS_FOR_OUTLIER_CHECK and run_len < MIN_STABLE_RUN_LEN
            else 0.0
        )

        return (
            (n * 100.0)
            + (run_len * 5.0)
            - (outliers * 30.0)
            - (small_cnt * 20.0)
            - unstable_penalty
            - (threshold * 0.1)
        )

    def _small_area_count(self, objects: list[DetectedObject]) -> int:
        """計算小面積物件數量"""
        if not objects:
            return 0
        areas = [o.area_px for o in objects]
        med = max(1, self._median_int(areas))
        cutoff = med * 0.25
        return sum(1 for a in areas if a < cutoff)
