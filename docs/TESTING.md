# 測試文檔

## 概述

本專案包含完整的測試套件，涵蓋單元測試 (Unit Tests) 和端到端測試 (E2E Tests)。

## 測試結構

```
tests/
├── __init__.py
├── conftest.py              # Pytest 配置和共用 fixtures
├── test_ultra_backend.py    # Ultra Backend 單元測試
├── test_unified_backend.py  # Unified Backend 單元測試
├── test_core.py             # 核心模組測試
└── test_e2e.py              # 端到端測試
```

## 測試圖片

測試會自動生成以下測試圖片：

1. **simple_portrait.png** - 簡單人像（白底圓形臉）
2. **greenscreen.png** - 綠幕背景
3. **white_background.png** - 純白背景（產品攝影風格）
4. **black_background.png** - 純黑背景（工作室風格）
5. **complex_edges.png** - 複雜邊緣（模擬頭髮、毛髮）
6. **gradient_background.png** - 漸層背景（困難場景）

## 運行測試

### 安裝依賴

```bash
uv sync
```

### 運行所有測試（快速，跳過慢速測試）

```bash
pytest -m "not slow"
```

### 運行所有測試（包含慢速測試）

```bash
pytest
```

### 運行特定類型的測試

```bash
# 只運行單元測試
pytest -m unit

# 只運行整合測試
pytest -m integration

# 只運行端到端測試
pytest -m e2e

# 運行快速測試（排除慢速）
pytest -m "not slow"
```

### 運行特定文件的測試

```bash
# Ultra Backend 測試
pytest tests/test_ultra_backend.py

# 端到端測試
pytest tests/test_e2e.py

# 核心模組測試
pytest tests/test_core.py
```

### 運行特定測試

```bash
# 運行特定測試類
pytest tests/test_ultra_backend.py::TestUltraBackendInit

# 運行特定測試函數
pytest tests/test_ultra_backend.py::TestUltraBackendInit::test_default_init
```

### 查看詳細輸出

```bash
pytest -v
```

### 查看測試覆蓋率

```bash
pytest --cov=src --cov-report=html
```

然後打開 `htmlcov/index.html` 查看詳細覆蓋率報告。

## 測試標記 (Markers)

### 內建標記

- **`slow`** - 慢速測試（需要載入模型或處理圖片）
  ```bash
  # 跳過慢速測試
  pytest -m "not slow"

  # 只運行慢速測試
  pytest -m slow
  ```

- **`integration`** - 整合測試（測試多個組件互動）
  ```bash
  pytest -m integration
  ```

- **`e2e`** - 端到端測試（測試完整流程）
  ```bash
  pytest -m e2e
  ```

- **`comparison`** - 比較測試（比較不同後端）
  ```bash
  pytest -m comparison
  ```

### 組合使用

```bash
# 運行整合測試但排除慢速
pytest -m "integration and not slow"

# 運行端到端測試且為慢速
pytest -m "e2e and slow"
```

## 測試覆蓋範圍

### Ultra Backend (`test_ultra_backend.py`)

- ✅ 初始化和配置
- ✅ 強度限制
- ✅ 色彩過濾配置
- ✅ 設備選擇
- ✅ 模型載入
- ✅ Trimap 建立
- ✅ 進階 Defringing
- ✅ 色彩過濾（綠/白/黑）
- ✅ 完整處理流程
- ✅ 錯誤處理

### Unified Backend (`test_unified_backend.py`)

- ✅ 基本初始化
- ✅ 圖片處理
- ✅ 色彩過濾
- ✅ 類方法

### 核心模組 (`test_core.py`)

- ✅ ProcessConfig
- ✅ ProcessResult
- ✅ ImageFile
- ✅ 圖片格式支援
- ✅ ImageProcessor
- ✅ 資料夾掃描

### 端到端測試 (`test_e2e.py`)

- ✅ 簡單人像處理
- ✅ 綠幕背景移除
- ✅ 純白/黑背景移除
- ✅ 複雜邊緣處理
- ✅ 漸層背景處理
- ✅ 批次處理
- ✅ ImageProcessor 整合
- ✅ Unified vs Ultra 比較
- ✅ 錯誤恢復
- ✅ 不同強度測試

## 持續整合 (CI)

建議在 CI 環境中運行：

```yaml
# .github/workflows/test.yml 示例
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run fast tests
        run: pytest -m "not slow"
      - name: Run slow tests (on main branch only)
        if: github.ref == 'refs/heads/main'
        run: pytest -m slow
```

## 效能考量

### 慢速測試

慢速測試（標記為 `slow`）主要包括：

1. **模型載入** - 首次載入需要下載模型（~500MB-1GB）
2. **圖片處理** - 實際運行 AI 模型進行推論
3. **批次處理** - 處理多張圖片

**建議**：
- 本地開發時使用 `-m "not slow"` 跳過
- 提交前運行完整測試套件
- CI 環境中分階段運行（快速測試 + 慢速測試）

### 記憶體使用

- Ultra Backend 測試需要 **4-6GB VRAM** (GPU) 或 **8-12GB RAM** (CPU)
- 如果記憶體不足，測試會自動跳過 (pytest.skip)

## 故障排除

### 問題：測試跳過 (skipped)

**原因**：模型載入失敗或網路問題

**解決方案**：
1. 檢查網路連接
2. 手動預先下載模型：
   ```bash
   python -c "from src.backends.ultra import UltraBackend; b = UltraBackend(); b.load_model()"
   ```
3. 檢查 Hugging Face 是否可訪問

### 問題：CUDA out of memory

**原因**：GPU 記憶體不足

**解決方案**：
1. 使用 CPU 運行測試：
   ```bash
   CUDA_VISIBLE_DEVICES="" pytest
   ```
2. 減少批次測試的圖片數量

### 問題：測試很慢

**原因**：運行慢速測試或使用 CPU

**解決方案**：
1. 跳過慢速測試：
   ```bash
   pytest -m "not slow"
   ```
2. 確保使用 GPU：
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## 添加新測試

### 單元測試範例

```python
# tests/test_my_feature.py
import pytest
from src.my_module import MyClass

class TestMyClass:
    """測試 MyClass"""

    def test_initialization(self) -> None:
        """測試初始化"""
        obj = MyClass(param=1)
        assert obj.param == 1

    @pytest.mark.slow
    def test_heavy_operation(self) -> None:
        """測試耗時操作"""
        obj = MyClass()
        result = obj.heavy_operation()
        assert result is not None
```

### 使用 Fixtures

```python
def test_with_image(simple_portrait_image: Path) -> None:
    """使用預先生成的測試圖片"""
    assert simple_portrait_image.exists()
    # 使用 simple_portrait_image 進行測試
```

### 創建自訂 Fixture

```python
# tests/conftest.py
@pytest.fixture
def my_custom_fixture(tmp_path: Path) -> Path:
    """創建自訂 fixture"""
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    return custom_dir
```

## 測試最佳實踐

1. **命名清晰**：測試函數名應清楚描述測試內容
   ```python
   def test_color_filter_removes_green_background() -> None:
       # 清楚知道這個測試在做什麼
   ```

2. **一個測試一個斷言**：盡可能一個測試只驗證一件事
   ```python
   # 好
   def test_strength_clamping_low() -> None:
       backend = UltraBackend(strength=0.05)
       assert backend.strength == 0.1

   def test_strength_clamping_high() -> None:
       backend = UltraBackend(strength=1.5)
       assert backend.strength == 1.0

   # 不好（兩個測試寫在一起）
   def test_strength_clamping() -> None:
       backend1 = UltraBackend(strength=0.05)
       assert backend1.strength == 0.1
       backend2 = UltraBackend(strength=1.5)
       assert backend2.strength == 1.0
   ```

3. **使用標記**：適當標記測試類型
   ```python
   @pytest.mark.slow
   @pytest.mark.integration
   def test_full_pipeline() -> None:
       pass
   ```

4. **清理資源**：使用 fixtures 自動清理
   ```python
   @pytest.fixture
   def temp_file(tmp_path: Path) -> Path:
       file = tmp_path / "test.txt"
       file.write_text("test")
       yield file
       # 自動清理（如果需要）
   ```

5. **跳過不適用的測試**：
   ```python
   @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
   def test_gpu_acceleration() -> None:
       pass
   ```

## 測試報告

生成 HTML 覆蓋率報告：

```bash
pytest --cov=src --cov-report=html
```

生成 XML 覆蓋率報告（用於 CI）：

```bash
pytest --cov=src --cov-report=xml
```

## 參考資源

- [Pytest 官方文檔](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**更新日期**: 2026-02-08
**版本**: 1.0.0
