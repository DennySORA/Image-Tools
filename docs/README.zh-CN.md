[English](../README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

# Image-Tools

一套全面的、生产级的图像处理工具套件，采用最先进的 AI 技术。提供专业级背景移除、水印移除和智能图像分割功能，完全本地运算，保护您的隐私。

## ✨ 主要功能

### 🎯 背景移除
- **Ultra 后端** - 使用 BRIA RMBG-2.0 达到最高质量（非商业用途）
- **Unified 后端** - 使用 BiRefNet-massive 适合商业用途（MIT 授权）
- **进阶处理** - Trimap 精细化、多色彩空间去边
- **颜色优化** - 针对绿色/白色/黑色背景的特殊处理
- **可控强度** - 精细调整移除强度（0.1-1.0）

### 🌊 水印移除
- **Gemini 水印移除器** - 移除 Google Gemini AI 生成的水印
- **反向 Alpha 混合** - 数学精确移除
- **自动检测** - 自动检测水印大小（48px/96px）
- **可调强度** - 部分或完全移除

### ✂️ 图像分割
- **智能网格布局** - 自动计算最佳网格配置
- **自定义排列** - 灵活的行×列配置
- **质量保留** - 无压缩、无损分割
- **批处理** - 一次分割多张图片

## 🚀 快速开始

### 安装

需求：Python 3.13+ 和 `uv` 包管理器

```bash
git clone https://github.com/yourusername/Image-Tools.git
cd Image-Tools
uv sync
```

### 运行

```bash
uv run main.py
```

按照交互式提示操作：
1. **选择文件夹** - 选择输入目录
2. **选择操作** - 背景移除、水印移除或图像分割
3. **配置设置** - 根据您的需求调整参数
4. **处理** - 让工具处理剩下的工作

输出文件将保存在 `<输入文件夹>/output/`

## 📚 功能深入介绍

### 背景移除后端选择

| 后端 | 质量 | 速度 | 授权 | 最适合 |
|---------|---------|-------|---------|----------|
| **Ultra** ⭐⭐⭐⭐⭐ | 极致 | 中等 | **非商业用途** | 个人项目、最高质量 |
| **Unified** ⭐⭐⭐⭐ | 优秀 | 快速 | **MIT（商业可用）** | 商业项目、生产环境 |

### 推荐设置

| 场景 | 强度 | 颜色过滤器 |
|----------|----------|--------------|
| 一般照片 | 0.5-0.7 | 关闭 |
| 复杂边缘（头发/毛发） | 0.6-0.8 | 关闭 |
| 绿幕 | 0.7-0.9 | 绿色 |
| 白色背景（产品） | 0.7-0.9 | 白色 |
| 黑色背景（工作室） | 0.7-0.9 | 黑色 |

## 🔧 技术细节

### 性能

- **单张图像**：2-5 秒（取决于分辨率和硬件）
- **内存使用**：~3-4GB GPU（BiRefNet-massive）、~2GB CPU
- **批处理**：高效的 session 重用
- **质量**：与商业服务相当或更好

### 硬件需求

| 配置 | 规格 |
|---------------|-------|
| **最低** | 8GB RAM + CPU（较慢） |
| **推荐** | 16GB RAM + 4GB+ VRAM GPU（CUDA） |
| **最佳** | 32GB RAM + 8GB+ VRAM GPU |

## 🆚 对比

| 功能 | 商业工具 | **Image-Tools** |
|---------|------------------|-----------------|
| 隐私 | ❌ 需要上传 | ✅ 100% 本地 |
| 自定义 | ❌ 有限控制 | ✅ 完全控制 |
| 成本 | 💰 $0.20+/图 | ✅ 免费（仅硬件） |
| 质量 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 功能 | 仅背景移除 | ✅ 多工具套件 |
| 授权 | 专有 | ✅ MIT（主要）、CC BY-NC（Ultra） |

## 🙏 致谢

本项目建立在优秀的开源作品之上：

- **BiRefNet** by [ZhengPeng7](https://github.com/ZhengPeng7/BiRefNet) - SOTA 分割
- **BRIA RMBG-2.0** by [BRIA AI](https://huggingface.co/briaai/RMBG-2.0) - 专业级模型
- **Rembg** by [danielgatis](https://github.com/danielgatis/rembg) - 背景移除工具包
- **Gemini Watermark Remover** by [journey-ad](https://github.com/journey-ad/gemini-watermark-remover) - 水印移除算法
- **Cloudflare** - 用于[评估和推荐 BiRefNet](https://blog.cloudflare.com/background-removal-with-workers-ai/)

## 📄 授权

**MIT 授权** - 详见 [LICENSE](../LICENSE)

**完全允许且鼓励商业使用**（Ultra 后端除外，需要非商业授权）。

### Ultra 后端授权

Ultra 后端使用的 BRIA RMBG-2.0 采用 **CC BY-NC 4.0**（非商业）授权。如果您计划商业使用 Ultra 后端，请联系 BRIA AI 获取授权。

---

**用 ❤️ 为开源社区制作**
