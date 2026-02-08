# 背景移除處理流程文檔

## 概述

Image-Tools 的背景移除功能使用多階段處理流程，基於 BRIA RMBG-2.0 模型，結合先進的邊緣處理技術，提供專業級的背景移除效果。

---

## 完整處理流程

```
輸入圖片
   ↓
[階段 1] RMBG-2.0 主分割 (Segmentation)
   ├─ 動態解析度處理 (1024/1536/2048/adaptive)
   ├─ Alpha Matte 生成 (0-255 連續透明度)
   └─ 高品質 Resize 回原圖 (CUBIC 插值)
   ↓
[階段 2] Trimap Refinement (可選)
   ├─ 建立 Trimap (腐蝕/膨脹)
   ├─ 識別確定前景/背景/未知區
   ├─ 未知區導向濾波 (Guided Filter)
   └─ 邊緣柔化 (Gaussian Blur)
   ↓
[階段 3] Advanced Defringing
   ├─ RGB 色彩平衡 (減少色偏)
   ├─ LAB 色彩空間分析 (a/b 通道去色)
   └─ 極半透明區域灰階混合
   ↓
[階段 4] Color Filter (可選)
   ├─ 綠幕：HSV 檢測 + Despill
   ├─ 黑色背景：LAB L 通道檢測
   └─ 白色背景：LAB L 通道檢測
   ↓
[階段 5] Edge Decontamination (新增 ✨)
   ├─ 識別半透明邊緣 (0.01 < alpha < 0.99)
   ├─ 估計背景色 (邊緣外圍像素)
   ├─ Unpremultiply-like 前景色還原
   └─ 修復白邊/綠邊/色彩污染
   ↓
[階段 6] Alpha Mode Processing (新增 ✨)
   ├─ Straight Alpha (預設)
   └─ Premultiplied Alpha (可選)
   ↓
[輸出] RGBA PNG
```

---

## 核心修復說明

### ✅ 修復 1: Premultiplied Alpha 處理

#### 問題根源
合成 RGBA 圖片到新背景時，如果 alpha 通道處理不當，會出現：
- **亮邊**：背景比前景亮時，邊緣出現白色光暈
- **暗邊**：背景比前景暗時，邊緣出現黑色陰影
- **色邊**：邊緣殘留原背景顏色（白邊、綠邊等）

#### 解決方案

**1. 邊緣去污染（Edge Decontamination）**
```python
decontaminate_edges(image, alpha, strength=0.7)
```

原理：
- 半透明邊緣像素 = 前景色 × alpha + 背景色 × (1-alpha)
- 估計背景色（使用邊緣外圍像素的中位數）
- 還原純前景色：`foreground = (pixel - bg × (1-alpha)) / alpha`
- 只在邊緣區域應用，不影響其他部分

**2. Alpha 模式支持**

| 模式 | 說明 | 適用場景 |
|------|------|---------|
| **Straight Alpha** | RGB 與 alpha 分離存儲（預設） | 一般圖片編輯、合成 |
| **Premultiplied Alpha** | RGB 已乘以 alpha | 特定渲染引擎、視頻合成 |

---

### ✅ 修復 2: 解析度處理改進

#### 問題根源
原實現固定使用 1024x1024 推論：
- **小圖（< 512）**：被放大，引入模糊
- **大圖（> 2048）**：被縮小，損失細節
- **邊界階梯**：低解析度上採樣放大瑕疵

#### 解決方案

**解析度模式**

| 模式 | 推論尺寸 | 適用場景 | 品質 | 速度 | 顯存 |
|------|---------|---------|------|------|------|
| `1024` | 1024×1024 | 一般圖片（預設） | ⭐⭐⭐ | ⚡⚡⚡ | 💾 |
| `1536` | 1536×1536 | 高品質需求 | ⭐⭐⭐⭐ | ⚡⚡ | 💾💾 |
| `2048` | 2048×2048 | 極致品質 | ⭐⭐⭐⭐⭐ | ⚡ | 💾💾💾 |
| `adaptive` | 自動選擇 | 混合場景 | ⭐⭐⭐⭐ | ⚡⚡ | 💾💾 |

**自適應解析度邏輯**

```python
if max_dim < 512:
    # 小圖：放大到至少 512
    scale = 512 / max_dim

elif max_dim < 1536:
    # 中圖：保持原樣
    use_original_size

elif max_dim > 2048:
    # 大圖：縮小到 2048
    scale = 2048 / max_dim
```

**插值改進**
- 原：`cv2.INTER_LINEAR`（快但品質一般）
- 新：`cv2.INTER_CUBIC`（慢一點但邊緣更平滑）

---

## 配置參數詳解

### AlphaConfig（Alpha 處理配置）

```python
AlphaConfig(
    mode=AlphaMode.STRAIGHT,           # Alpha 模式
    edge_decontamination=True,          # 啟用邊緣去污染
    decontamination_strength=0.7        # 去污染強度 (0.0-1.0)
)
```

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `mode` | AlphaMode | `STRAIGHT` | `STRAIGHT` 或 `PREMULTIPLIED` |
| `edge_decontamination` | bool | `True` | 是否啟用邊緣去污染 |
| `decontamination_strength` | float | `0.7` | 去污染強度，越高越激進 |

### ResolutionConfig（解析度配置）

```python
ResolutionConfig(
    mode=ResolutionMode.FIXED_1024,    # 解析度模式
    max_size=2048,                      # 自適應模式最大尺寸
    maintain_aspect_ratio=True          # 保持寬高比
)
```

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `mode` | ResolutionMode | `FIXED_1024` | 解析度模式 |
| `max_size` | int | `2048` | adaptive 模式的最大尺寸 |
| `maintain_aspect_ratio` | bool | `True` | 是否保持寬高比 |

---

## 使用範例

### 基礎使用（預設配置）

```python
from src.features.background_removal.ultra import UltraBackend

backend = UltraBackend(
    strength=0.7,  # 處理強度
)

backend.process(input_path, output_path)
```

### 高品質模式（2048 解析度 + 強力去污染）

```python
from src.common import AlphaConfig, ResolutionConfig, ResolutionMode

backend = UltraBackend(
    strength=0.8,
    resolution_config=ResolutionConfig(
        mode=ResolutionMode.FIXED_2048
    ),
    alpha_config=AlphaConfig(
        edge_decontamination=True,
        decontamination_strength=0.9
    )
)
```

### 綠幕背景（色彩過濾 + 去污染）

```python
from src.common import ColorFilterConfig, ColorFilter, AlphaConfig

backend = UltraBackend(
    strength=0.8,
    color_filter=ColorFilterConfig(
        enabled=True,
        color=ColorFilter.GREEN
    ),
    alpha_config=AlphaConfig(
        edge_decontamination=True,
        decontamination_strength=0.8
    )
)
```

### 自適應解析度（智能選擇）

```python
from src.common import ResolutionConfig, ResolutionMode

backend = UltraBackend(
    strength=0.7,
    resolution_config=ResolutionConfig(
        mode=ResolutionMode.ADAPTIVE,
        max_size=2048  # 最大不超過 2048
    )
)
```

---

## 技術細節

### 邊緣去污染算法

**第一步：識別邊緣區域**
```python
edge_mask = (alpha > 0.01) & (alpha < 0.99)
```

**第二步：估計背景色**
```python
# 擴展邊緣遮罩
dilated_edge = cv2.dilate(edge_mask, kernel, iterations=2)
background_region = (dilated_edge > 0) & (alpha < 0.1)

# 使用中位數（比平均值更穩健）
bg_color = np.median(image[background_region], axis=0)
```

**第三步：前景色還原**
```python
# 假設：pixel = foreground × alpha + background × (1-alpha)
# 求解：foreground = (pixel - bg × (1-alpha)) / alpha

foreground = (pixel - bg_color × (1 - alpha)) / alpha
```

**第四步：只替換邊緣區域**
```python
result[edge_mask] = foreground_estimate[edge_mask]
```

### Alpha 模式轉換

**Premultiply（預乘）**
```python
RGB_premul = RGB × (alpha / 255)
```

**Unpremultiply（反預乘）**
```python
RGB_straight = RGB_premul / (alpha / 255)  # 當 alpha > 0
```

---

## 效果對比

### 修復前 vs 修復後

| 場景 | 修復前問題 | 修復後效果 |
|------|-----------|-----------|
| **白色背景** | 邊緣有明顯白邊 | 邊緣乾淨，無白邊 |
| **綠幕背景** | 邊緣殘留綠色 | 綠色完全移除 |
| **複雜邊緣（頭髮）** | 鋸齒狀、階梯狀 | 平滑、自然過渡 |
| **小圖（< 512）** | 放大後模糊 | 保持清晰 |
| **大圖（> 2048）** | 細節損失 | 細節保留 |
| **合成到新背景** | 亮邊/暗邊 | 邊緣自然融合 |

---

## 性能指標

### 處理速度（參考）

| 解析度模式 | 1920×1080 | 3840×2160 | 7680×4320 |
|-----------|----------|----------|----------|
| **1024** | ~2s | ~2.5s | ~3s |
| **1536** | ~4s | ~5s | ~6s |
| **2048** | ~7s | ~9s | ~12s |
| **adaptive** | ~2-6s | ~4-8s | ~6-12s |

*測試環境：NVIDIA RTX 3090, CUDA 12.1*

### 顯存使用

| 解析度模式 | 顯存峰值 |
|-----------|---------|
| **1024** | ~2GB |
| **1536** | ~4GB |
| **2048** | ~7GB |
| **adaptive** | ~2-7GB |

---

## 常見問題

### Q1: 何時使用 Premultiplied Alpha？

**A:**
- ✅ **使用**：視頻編輯軟件（AE、Premiere）、遊戲引擎（Unity、Unreal）
- ❌ **不使用**：一般圖片編輯（Photoshop、GIMP）、網頁顯示

### Q2: 邊緣還是有一點點色邊，怎麼辦？

**A:** 提高去污染強度：
```python
alpha_config=AlphaConfig(
    decontamination_strength=0.9  # 更激進（預設 0.7）
)
```

### Q3: 處理速度太慢？

**A:** 降低解析度模式：
```python
resolution_config=ResolutionConfig(
    mode=ResolutionMode.FIXED_1024  # 最快
)
```

### Q4: 極致品質需求？

**A:** 使用最高配置：
```python
UltraBackend(
    strength=0.9,
    resolution_config=ResolutionConfig(
        mode=ResolutionMode.FIXED_2048
    ),
    alpha_config=AlphaConfig(
        decontamination_strength=0.9
    ),
    use_trimap_refine=True
)
```

---

## 技術參考

### 相關文獻

1. **BRIA RMBG-2.0**
   - 模型：https://huggingface.co/briaai/RMBG-2.0
   - 授權：CC BY-NC 4.0（非商用）

2. **BiRefNet**
   - 論文：Bilateral Reference for High-Resolution Dichotomous Image Segmentation
   - GitHub：https://github.com/ZhengPeng7/BiRefNet

3. **Alpha Matting**
   - 經典問題：前景/背景分離與透明度估計
   - 參考：https://www.alphamatting.com/

### 關鍵概念

- **Dichotomous Segmentation**：二分圖像分割（前景/背景）
- **Alpha Matte**：連續透明度圖（0-1 或 0-255）
- **Trimap**：三值圖（確定前景/確定背景/未知區）
- **Defringing**：去除邊緣色彩污染
- **Premultiplied Alpha**：RGB 預乘 alpha 的存儲格式

---

## 更新日誌

### v1.1.0 (2026-02-08) ✨

**核心修復**
- ✅ 添加邊緣去污染（Edge Decontamination）
- ✅ 支持多種解析度模式（1024/1536/2048/adaptive）
- ✅ 支持 Premultiplied/Straight Alpha 輸出
- ✅ 改進插值算法（LINEAR → CUBIC）

**技術改進**
- 新增 `src/common/alpha_config.py` 配置模組
- 動態解析度推論
- 智能背景色估計算法

**效果提升**
- 邊緣白邊/綠邊/色邊問題大幅改善
- 小圖/大圖處理品質提升
- 合成到新背景更自然

---

## 授權聲明

**⚠️ 重要提醒**

BRIA RMBG-2.0 模型採用 **CC BY-NC 4.0** 授權：
- ✅ **允許**：個人使用、研究、教育用途
- ❌ **禁止**：商業使用（需與 BRIA 另行協議）

商業使用請聯繫：https://bria.ai/

---

**文檔版本**：v1.1.0
**最後更新**：2026-02-08
**維護者**：Image-Tools Team
