[繁體中文](docs/README.zh-TW.md) | [简体中文](docs/README.zh-CN.md) | [日本語](docs/README.ja.md)

# Remove Background

A powerful, production-grade tool for removing image backgrounds using state-of-the-art AI technology. Features a unified SOTA backend with multi-stage processing for professional-quality edge handling.

## ✨ Highlights

🏆 **Ultra Backend (NEW!)** - Maximum quality with BRIA RMBG-2.0 (non-commercial)
🎯 **Unified SOTA Backend** - Best-in-class with BiRefNet-massive (MIT, commercial OK)
🔬 **Advanced Processing** - Trimap Refinement → Multi-space Defringing
🎨 **Pure Color Optimization** - Special handling for green/white/black backgrounds
💪 **Controllable Strength** - Fine-tune removal aggressiveness (0.1-1.0)
🚀 **Batch Processing** - Handle entire folders efficiently
🔒 **100% Local** - No API calls, complete privacy
📦 **Flexible Licensing** - MIT (commercial) or CC BY-NC (ultimate quality)

## Installation

Requirements: Python 3.13+ and `uv` package manager

```bash
git clone <repository-url>
cd Remove-Background
uv sync
```

## Quick Start

```bash
uv run main.py
```

Follow the interactive prompts:
1. **Select Folder** - Choose input directory
2. **Select Operation** - Background removal, watermark removal, or image splitting
3. **Configure Settings** - Adjust strength and optional color filter
4. **Process** - Sit back while the tool processes all images

Output images are saved as transparent PNGs in `<input-folder>/output/`

## 🎯 Background Removal Backends

### Which Backend to Use?

| Backend | Quality | Speed | License | Best For |
|---------|---------|-------|---------|----------|
| **Ultra** ⭐⭐⭐⭐⭐ | Extreme | Medium | **Non-commercial** | Personal use, maximum quality |
| **Unified** ⭐⭐⭐⭐ | Excellent | Fast | **MIT (commercial OK)** | Commercial projects, general use |

### Ultra Backend (Maximum Quality)

**⚠️ Non-commercial use only (CC BY-NC 4.0)**

The ultimate solution for personal users seeking the absolute best quality:

#### Features
- **BRIA RMBG-2.0**: Professional-grade model with superior training data
- **Trimap Refinement**: Intelligent boundary processing (preserves details)
- **Multi-space Defringing**: RGB + LAB + HSV color analysis
- **Guided Filter**: Edge-aware smoothing (better than Gaussian blur)
- **Color Filter** (Optional): Optimized for pure-color backgrounds

[📚 Full Ultra Backend Documentation](docs/ULTRA_BACKEND.md)

---

### Unified Backend (Commercial-Friendly)

**✅ MIT License - Commercial use OK**

Balanced solution combining quality and flexibility:

#### Features
- **BiRefNet-massive**: SOTA segmentation with excellent edge preservation
- **Alpha Matting**: Automatic refinement for complex edges
- **Edge Defringing**: Removes color contamination
- **Color Filter** (Optional): Pure-color background optimization

[📚 Full Unified Backend Documentation](docs/UNIFIED_BACKEND.md)

### Recommended Settings

| Scenario | Strength | Color Filter |
|----------|----------|--------------|
| General photos | 0.5-0.7 | Off |
| Complex edges (hair/fur) | 0.6-0.8 | Off |
| Green screen | 0.7-0.9 | Green |
| White background (product) | 0.7-0.9 | White |
| Black background (studio) | 0.7-0.9 | Black |

### When to Enable Color Filter?

✅ **Enable** for:
- Green screen photography/video
- Product photography (pure white/black background)
- ID photos (solid color background)

❌ **Disable** for:
- Natural scenes (complex backgrounds)
- Gradient backgrounds
- Mixed backgrounds

## Key Features

- **State-of-the-Art Quality**: BiRefNet-massive model outperforms traditional methods
- **Smart Edge Handling**: Alpha matting preserves fine details like hair and fur
- **No More Fringing**: Automatic edge decontamination removes color spill
- **Batch Processing**: Process entire folders with progress feedback
- **Flexible Control**: Adjustable strength from conservative to aggressive
- **Multiple Operations**: Background removal, watermark removal, image splitting
- **User-Friendly UI**: Modern interactive CLI with ESC navigation support

## 📚 Documentation

- **[🏆 Ultra Backend Guide](docs/ULTRA_BACKEND.md)** - Maximum quality (non-commercial)
- **[Unified Backend Guide](docs/UNIFIED_BACKEND.md)** - Commercial-friendly solution
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Upgrading from old backends
- **[ESC Key Navigation Fix](docs/ESC_KEY_FIX.md)** - Troubleshooting UI navigation

## 🔧 Technical Details

### Processing Pipeline

```
Input Image
    ↓
Stage 1: BiRefNet Segmentation
    ├─ High-resolution dichotomous segmentation
    ├─ Dynamic alpha matting (auto-enabled at strength ≥ 0.3)
    └─ Adaptive foreground/background thresholds
    ↓
Stage 2: Color Filter (Optional)
    ├─ HSV/LAB color space detection
    ├─ Morphological mask refinement
    ├─ Edge despill (for green screens)
    └─ Alpha channel merging
    ↓
Stage 3: Edge Defringing
    ├─ Half-transparent edge detection
    ├─ Color imbalance analysis
    └─ Edge color correction
    ↓
Output PNG (RGBA)
```

### Performance

- **Single Image**: 2-5 seconds (depends on resolution and hardware)
- **Memory Usage**: ~3-4GB GPU (BiRefNet-massive)
- **Batch Processing**: Efficient session reuse
- **Quality**: Comparable to or better than commercial services (e.g., remove.bg)

### Hardware Requirements

| Configuration | Specs |
|---------------|-------|
| **Minimum** | 8GB RAM + CPU (slower) |
| **Recommended** | 16GB RAM + 4GB+ VRAM GPU (CUDA) |
| **Optimal** | 32GB RAM + 8GB+ VRAM GPU |

## 🆚 Comparison

| Feature | remove.bg | **Ultra Backend** | Unified Backend |
|---------|-----------|-------------------|-----------------|
| License | Commercial (paid) | **Non-commercial (free)** | MIT (free, commercial) |
| Quality | Excellent | **⭐⭐⭐⭐⭐ Extreme** | ⭐⭐⭐⭐ Excellent |
| Edge Detail | Good | **⭐⭐⭐⭐⭐ Extreme** | ⭐⭐⭐⭐ Excellent |
| Processing | RGB + LAB | **RGB + LAB + HSV** | RGB + LAB |
| Trimap Refinement | ❌ | **✅ Complete** | ❌ |
| Privacy | ❌ Upload required | **✅ 100% local** | ✅ 100% local |
| Customization | ❌ No control | **✅ Full control** | ✅ Strength + filter |
| Cost | $0.20+/image | **Free (hardware)** | Free (hardware) |
| Best For | Quick API use | **Personal, max quality** | Commercial projects |

## External Dependencies (Third-Party)

- **rembg** (MIT) — Includes BiRefNet models for background removal
- **transparent-background** (MIT) — Alternative background removal option
- **backgroundremover** (MIT) — Alternative background removal option
- **onnxruntime** (MIT) — Runtime support for ONNX models
- **pillow** (HPND) — Image reading and writing utilities
- **opencv-python** (Apache 2.0) — Computer vision operations
- **InquirerPy** (MIT) — Interactive CLI interface
- **moviepy** (MIT) — Media utility library

All dependencies are open-source with permissive licenses.

## 🙏 Acknowledgments

This project builds upon excellent open-source work:

- **BiRefNet** by [ZhengPeng7](https://github.com/ZhengPeng7/BiRefNet) - SOTA segmentation
- **Rembg** by [danielgatis](https://github.com/danielgatis/rembg) - Background removal toolkit
- **Cloudflare** - For [evaluating and recommending BiRefNet](https://blog.cloudflare.com/background-removal-with-workers-ai/)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

**Commercial use is fully allowed and encouraged.**
