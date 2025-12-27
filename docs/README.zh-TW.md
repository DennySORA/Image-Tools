[English](../README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

# Remove Background

這是一款易用的圖片去背工具，提供多種方案與引導式流程，適合批次處理，也能應付一般照片與綠幕照片。

## 安裝
- 確認 Python 3.13 與 `uv` 可用。
- 下載專案並安裝相依套件。

```bash
git clone <repository-url>
cd Remove-Background
uv sync
```

## 使用方法
- 啟動互動流程並依提示操作。
- 輸出為透明 PNG，存放在所選資料夾的 `output/` 內。

```bash
uv run main.py
```

## 使用方案
- 引導式流程：選擇資料夾、選擇方案、調整強度、確認後開始處理。
- 綠幕流程：適合純綠背景，讓邊緣更乾淨並減少色彩殘留。
- 一般照片流程：適用人像、商品與混合背景的照片。
- 速度與品質取捨：可依時間與成品質感選擇模式。

## 功能介紹
- 多種去背方案與清楚的說明。
- 批次處理，並提供進度與結果摘要。
- 可調整強度，控制去背程度。
- 輸出透明背景，方便後續使用。

## 外部依賴（第三方）
- **backgroundremover** — 外部依賴。背景移除方案之一。
- **rembg** — 外部依賴。另一種背景移除方案。
- **transparent-background** — 外部依賴。背景移除方案。
- **onnxruntime** — 外部依賴。支援上述方案的執行環境。
- **pillow** — 外部依賴。影像讀寫工具。
- **moviepy** — 外部依賴。列於相依清單的媒體工具。
