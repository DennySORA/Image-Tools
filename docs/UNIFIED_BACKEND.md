# 統一背景移除後端 (Unified Backend)

## 概述

統一背景移除後端是基於最新研究和最佳實踐開發的 SOTA（State-of-the-Art）解決方案，整合了多階段處理流程以提供最佳的背景移除品質。

## 技術架構

### 核心技術棧

1. **BiRefNet-massive** - 主要分割模型
   - 來源：[Bilateral Reference for High-Resolution Dichotomous Image Segmentation (CAAI AIR 2024)](https://github.com/ZhengPeng7/BiRefNet)
   - 特點：高解析度分割、優秀的邊緣保持能力、動態解析度支援
   - 授權：MIT License（可商用）

2. **Alpha Matting** - 邊緣精細化
   - 使用 pymatting 算法進行半透明邊界處理
   - 特別適合頭髮、毛髮、透明材質等複雜邊緣

3. **色彩去污染 (Defringing)** - 邊緣顏色修正
   - 移除邊緣的顏色滲透（白邊、灰邊等）
   - 基於色彩平衡算法

4. **可選色彩過濾** - 純色背景優化
   - 針對綠幕、純白、純黑背景的專門優化
   - 結合色度鍵（Chroma Key）和形態學操作

## 處理流程

```
輸入圖片
    ↓
階段 1: BiRefNet 主要分割
    ├─ 使用 BiRefNet-massive 模型
    ├─ 自動啟用 Alpha Matting（強度 ≥ 0.3）
    └─ 動態調整前景/背景閾值
    ↓
階段 2: 可選色彩過濾（如果啟用）
    ├─ 根據背景顏色類型建立遮罩
    ├─ 使用形態學操作優化遮罩
    ├─ 邊緣 Despill 處理（針對綠幕）
    └─ 與主 Alpha 合併
    ↓
階段 3: 邊緣去污染
    ├─ 檢測半透明邊緣
    ├─ 分析色彩不平衡
    └─ 校正邊緣顏色
    ↓
輸出 PNG（RGBA）
```

## 參數說明

### 強度 (Strength): 0.1 - 1.0

| 範圍 | 模式 | 說明 | 適用場景 |
|------|------|------|----------|
| 0.1-0.3 | 保守模式 | Alpha Matting 關閉，保留更多邊緣像素 | 邊界清晰、對比度高的圖片 |
| 0.4-0.7 | 平衡模式 | Alpha Matting 啟用，平衡品質與完整性 | **一般推薦**，適合大多數情況 |
| 0.8-1.0 | 激進模式 | 最大強度處理，可能損失細節 | 背景複雜、需要徹底去除的情況 |

**推薦設定**：
- 一般圖片：0.5-0.7
- 複雜邊緣（頭髮、毛髮）：0.6-0.8
- 純色背景 + 色彩過濾：0.7-0.9

### 色彩過濾選項

#### 何時啟用？

當您的圖片背景是**純色**時，啟用色彩過濾可以顯著改善邊緣品質：

- ✅ 綠幕攝影/視訊
- ✅ 產品攝影（純白/純黑背景）
- ✅ 證件照（純色背景）
- ❌ 自然場景（複雜背景）
- ❌ 漸層背景

#### 支援的背景類型

1. **🟢 綠幕背景 (Green Screen)**
   - HSV 色度鍵檢測
   - 自動 Despill（移除綠色溢出）
   - 適用：綠幕攝影棚、Zoom 虛擬背景

2. **⚪ 純白背景 (White)**
   - LAB 色彩空間亮度檢測
   - 適用：產品攝影、電商圖片

3. **⚫ 純黑背景 (Black)**
   - LAB 色彩空間亮度檢測
   - 適用：工作室人像、藝術攝影

## 使用示例

### CLI 使用

```bash
# 啟動程式
uv run main.py

# 選擇流程：
# 1. 選擇資料夾
# 2. 選擇「背景移除」操作
# 3. 設定強度（例如：0.7）
# 4. 選擇是否啟用色彩過濾
#    - 如果啟用：選擇背景顏色（綠/白/黑）
```

### Python API 使用

```python
from pathlib import Path
from src.backends.unified import UnifiedBackend, ColorFilterConfig, ColorFilter

# 基本使用（無色彩過濾）
backend = UnifiedBackend(strength=0.7)
backend.load_model()
backend.process(
    input_path=Path("input.jpg"),
    output_path=Path("output.png")
)

# 使用綠幕過濾
color_filter = ColorFilterConfig(
    enabled=True,
    color=ColorFilter.GREEN,
    edge_refine_strength=0.7
)
backend = UnifiedBackend(strength=0.8, color_filter=color_filter)
backend.load_model()
backend.process(
    input_path=Path("greenscreen.jpg"),
    output_path=Path("output.png")
)
```

## 效能與品質

### 邊緣品質比較

與傳統方案相比：

| 特性 | 傳統方案 | 統一後端 |
|------|----------|----------|
| 頭髮邊緣 | ⚠️ 經常出現鋸齒 | ✅ 平滑、半透明 |
| 毛邊處理 | ⚠️ 粗糙、破碎 | ✅ 細節保留良好 |
| 透明材質 | ❌ 無法處理 | ✅ 支援半透明 |
| 邊緣顏色 | ⚠️ 常見白邊/綠邊 | ✅ 自動去污染 |
| 純色背景 | ⚠️ 仍有殘留 | ✅ 色彩過濾可完全移除 |

### 處理速度

- **單張圖片**：約 2-5 秒（取決於解析度和硬體）
- **批次處理**：支援，自動重用模型會話
- **記憶體使用**：約 3-4GB GPU（BiRefNet-massive）

### 硬體需求

- **最低**：8GB RAM + CPU（較慢）
- **推薦**：16GB RAM + 4GB+ VRAM GPU（CUDA）
- **最佳**：32GB RAM + 8GB+ VRAM GPU

## 常見問題 (FAQ)

### Q1: 為什麼邊緣還是有毛邊？

**可能原因**：
1. 強度設定過低（< 0.5）→ 提高到 0.6-0.8
2. 複雜背景但未啟用足夠強度 → 嘗試 0.7-0.9
3. 圖片解析度過低 → 使用高解析度原圖

### Q2: 邊緣處理後顏色變淡了？

**原因**：邊緣去污染過度

**解決方案**：
- 降低強度到 0.5-0.6
- 如果是綠幕，確保啟用了色彩過濾（綠）

### Q3: 純白/純黑背景還有殘留？

**解決方案**：
1. **啟用色彩過濾**並選擇對應顏色
2. 提高強度到 0.8-0.9
3. 確保背景確實是「純色」（均勻、無漸層）

### Q4: 處理速度太慢？

**優化建議**：
1. 確保有 GPU 並安裝 CUDA 版本的 PyTorch
2. 批次處理時重用模型（已自動實現）
3. 考慮降低圖片解析度（預處理）

### Q5: 相比其他工具（如 remove.bg）如何？

| 特性 | remove.bg | 統一後端 |
|------|-----------|----------|
| 授權 | 商業服務（付費） | MIT（免費、可商用） |
| 品質 | 優秀 | 相當或更好 |
| 邊緣細節 | 好 | 非常好（Alpha Matting） |
| 隱私 | ❌ 需上傳圖片 | ✅ 完全本地處理 |
| 客製化 | ❌ 無法調整 | ✅ 強度、色彩過濾可調 |

## 技術參考

### 相關論文與工具

1. **BiRefNet**: [GitHub](https://github.com/ZhengPeng7/BiRefNet) | [Hugging Face](https://huggingface.co/ZhengPeng7/BiRefNet)
2. **Rembg**: [GitHub](https://github.com/danielgatis/rembg) | [PyPI](https://pypi.org/project/rembg/)
3. **Alpha Matting**: [GitHub - pymatting](https://github.com/pymatting/pymatting)

### 外部評測

- **Cloudflare Blog**: [Why we picked BiRefNet for background removal](https://blog.cloudflare.com/background-removal-with-workers-ai/)

## 授權

統一後端基於以下開源項目：

- **BiRefNet**: MIT License
- **Rembg**: MIT License
- **本專案**: MIT License

✅ **完全可商用**，無需擔心授權問題。

---

**更新日期**: 2026-02-08
**版本**: 1.0.0
