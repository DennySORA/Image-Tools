# 預設配置使用指南

本專案提供三個簡化的預設配置等級，讓你無需手動調整大量參數。

## 預設等級對比

| 功能 | Balanced（中等） | High（高） | Ultra（最強） |
|------|----------------|-----------|-------------|
| **強度** | 0.6 | 0.75 | 0.95 |
| **解析度** | 1024px | 1536px | 2048px |
| **Trimap 精修** | ✓ | ✓ | ✓ |
| **人像精修** | ✗ | ✓ (Enhanced) | ✓ (MODNet) |
| **邊緣去污染** | ✓ (0.6) | ✓ (0.75) | ✓ (0.95) |
| **適用場景** | 日常使用 | 專業工作 | 極致品質 |
| **處理速度** | 快 | 中 | 慢 |

## 基本使用

### 方法 1: 使用預設配置（推薦）

```python
from src.features.background_removal.ultra import UltraBackend

# 中等預設（推薦日常使用）
backend = UltraBackend.from_preset("balanced")

# 高品質預設（推薦專業工作）
backend = UltraBackend.from_preset("high")

# 最強預設（追求極致效果）
backend = UltraBackend.from_preset("ultra")

# 處理圖片
backend.load_model()
backend.process("input.png", "output.png")
```

### 方法 2: 使用 PresetLevel 枚舉

```python
from src.common import PresetLevel
from src.features.background_removal.ultra import UltraBackend

backend = UltraBackend.from_preset(PresetLevel.HIGH)
```

## 進階使用

### 添加色彩過濾

針對純色背景（綠幕、黑背景、白背景）：

```python
from src.common import ColorFilter, ColorFilterConfig
from src.features.background_removal.ultra import UltraBackend

# 綠幕過濾
green_filter = ColorFilterConfig(
    color=ColorFilter.GREEN,
    enabled=True,
    edge_refine_strength=0.8
)

backend = UltraBackend.from_preset("high", color_filter=green_filter)

# 黑背景過濾
black_filter = ColorFilterConfig(color=ColorFilter.BLACK, enabled=True)
backend = UltraBackend.from_preset("balanced", color_filter=black_filter)

# 白背景過濾
white_filter = ColorFilterConfig(color=ColorFilter.WHITE, enabled=True)
backend = UltraBackend.from_preset("balanced", color_filter=white_filter)
```

### 指定計算設備

```python
# 強制使用 GPU
backend = UltraBackend.from_preset("ultra", device="cuda")

# 強制使用 CPU
backend = UltraBackend.from_preset("balanced", device="cpu")

# 自動選擇（預設）
backend = UltraBackend.from_preset("high")  # 自動檢測 CUDA
```

## 預設配置詳細說明

### Balanced（中等）

**適合場景**：
- 日常圖片處理
- 需要快速處理大量圖片
- 對品質要求不是特別高

**特點**：
- 強度 0.6（平衡去除與保留）
- 1024px 解析度（速度快）
- 啟用 Trimap 精修
- **不啟用**人像精修（節省時間）
- 邊緣去污染強度 0.6

**處理速度**：最快

### High（高）

**適合場景**：
- 專業攝影作品
- 需要高品質的人像處理
- 有頭髮/毛髮的複雜邊緣

**特點**：
- 強度 0.75（較激進但保留細節）
- 1536px 解析度（高品質）
- 啟用 Trimap 精修
- **啟用人像精修**（Enhanced mode）
- 邊緣去污染強度 0.75

**處理速度**：中等

### Ultra（最強）

**適合場景**：
- 追求極致品質
- 專業人像攝影
- 商業級產品照片
- 不在意處理時間

**特點**：
- 強度 0.95（最大去除）
- 2048px 解析度（極致品質）
- 啟用 Trimap 精修
- **啟用 MODNet 人像精修**（專業模型）
- 邊緣去污染強度 0.95

**處理速度**：最慢

**注意**：Ultra 預設會嘗試載入 MODNet 模型。如果 MODNet 不可用，會自動回退到 Enhanced mode。

## MODNet 專業模型

Ultra 預設使用 MODNet 專業人像 matting 模型。

### 安裝依賴

```bash
# 安裝 ONNX Runtime（推薦，速度快）
uv add onnxruntime

# 如果有 GPU，安裝 GPU 版本
uv add onnxruntime-gpu

# 安裝 Hugging Face Hub（用於下載模型）
uv add huggingface-hub
```

### MODNet 自動下載

首次使用 Ultra 預設時，MODNet 模型會自動從 Hugging Face 下載：

```python
backend = UltraBackend.from_preset("ultra")
backend.load_model()  # 首次執行會下載 MODNet
```

模型會快取在 `.cache/modnet/` 目錄，之後不需要重複下載。

### MODNet 回退機制

如果 MODNet 無法載入（例如依賴未安裝），系統會自動回退到 Enhanced mode（進階影像處理），不會中斷處理。

## 列出可用預設

```python
from src.features.background_removal.ultra import UltraBackend

presets = UltraBackend.get_available_presets()
for name, description in presets.items():
    print(f"{name}: {description}")
```

輸出：
```
balanced: 中等 - 平衡速度與品質（強度 0.6，1024px）
high: 高 - 高品質處理（強度 0.75，1536px，人像精修）
ultra: 最強 - 全部參數拉滿（強度 0.95，2048px，MODNet 人像精修）
```

## 列印對比表

```python
from src.common import print_preset_comparison

print_preset_comparison()
```

## 完整範例

```python
from pathlib import Path
from src.common import ColorFilter, ColorFilterConfig
from src.features.background_removal.ultra import UltraBackend

def process_with_preset(
    input_path: str,
    output_path: str,
    preset: str = "high",
    color_filter_type: str | None = None
):
    """
    使用預設配置處理圖片

    Args:
        input_path: 輸入圖片路徑
        output_path: 輸出圖片路徑
        preset: 預設等級（balanced/high/ultra）
        color_filter_type: 色彩過濾類型（green/black/white/None）
    """
    # 建立色彩過濾（如果指定）
    color_filter = None
    if color_filter_type:
        color_map = {
            "green": ColorFilter.GREEN,
            "black": ColorFilter.BLACK,
            "white": ColorFilter.WHITE,
        }
        color_filter = ColorFilterConfig(
            color=color_map[color_filter_type],
            enabled=True
        )

    # 建立後端
    backend = UltraBackend.from_preset(preset, color_filter=color_filter)

    # 載入模型並處理
    backend.load_model()
    success = backend.process(Path(input_path), Path(output_path))

    if success:
        print(f"✓ 處理成功: {output_path}")
    else:
        print(f"✗ 處理失敗: {input_path}")

    return success


# 使用範例
if __name__ == "__main__":
    # 一般圖片，使用 Balanced
    process_with_preset("photo1.jpg", "output1.png", "balanced")

    # 人像圖片，使用 High
    process_with_preset("portrait.jpg", "output2.png", "high")

    # 綠幕圖片，使用 High + 綠幕過濾
    process_with_preset("greenscreen.jpg", "output3.png", "high", "green")

    # 極致品質，使用 Ultra
    process_with_preset("professional.jpg", "output4.png", "ultra")
```

## 測試腳本

運行測試腳本查看所有預設配置的詳細資訊：

```bash
python test_presets.py
```

## 注意事項

1. **授權限制**：Ultra backend 使用 BRIA RMBG-2.0 模型（CC BY-NC 4.0），僅供非商用使用
2. **硬體需求**：
   - Balanced: 4GB+ RAM，可用 CPU
   - High: 8GB+ RAM，推薦 GPU
   - Ultra: 12GB+ RAM，強烈推薦 GPU（8GB+ VRAM）
3. **處理時間**：Ultra 預設處理時間約為 Balanced 的 3-5 倍
4. **MODNet 依賴**：Ultra 預設需要額外安裝 `onnxruntime` 和 `huggingface-hub`

## 常見問題

### Q: 如何選擇預設？

- **快速處理**：選擇 Balanced
- **人像照片**：選擇 High
- **極致品質**：選擇 Ultra

### Q: MODNet 模型下載失敗怎麼辦？

系統會自動回退到 Enhanced mode，不影響處理。如需使用 MODNet：
1. 確保網路正常
2. 安裝依賴：`uv add onnxruntime huggingface-hub`
3. 手動下載模型到 `.cache/modnet/`

### Q: 可以自訂預設嗎？

目前提供三個固定預設。如需更多控制，請直接使用 `UltraBackend.__init__()` 並手動設定參數。

### Q: 預設配置可以覆蓋嗎？

可以。使用 `from_preset()` 後，仍可覆蓋部分參數（如 `color_filter` 和 `device`）。
