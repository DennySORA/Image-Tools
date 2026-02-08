[繁體中文](docs/README.zh-TW.md) | [简体中文](docs/README.zh-CN.md) | [日本語](docs/README.ja.md)

# Image-Tools

A comprehensive, production-grade image processing toolkit powered by state-of-the-art AI technology. Perform professional background removal, watermark removal, and intelligent image splitting with complete privacy and control.

## ✨ Key Features

### 🎯 Background Removal
- **Ultra Backend** - Maximum quality with BRIA RMBG-2.0 (non-commercial)
- **Advanced Processing** - Trimap refinement, multi-space defringing
- **Color Optimization** - Special handling for green/white/black backgrounds
- **Controllable Strength** - Fine-tune removal aggressiveness (0.1-1.0)

### 🌊 Watermark Removal
- **Gemini Watermark Remover** - Remove Google Gemini AI watermarks
- **Reverse Alpha Blending** - Mathematical precision removal
- **Auto Detection** - Automatic watermark size detection (48px/96px)
- **Adjustable Strength** - Partial or complete removal

### ✂️ Image Splitting
- **Smart Grid Layout** - Automatic optimal grid calculation
- **Custom Arrangements** - Flexible rows × columns configuration
- **Quality Preservation** - No compression, lossless splitting
- **Batch Processing** - Split multiple images at once

## 🚀 Quick Start

### Installation

Requirements: Python 3.13+ and `uv` package manager

```bash
git clone https://github.com/yourusername/Image-Tools.git
cd Image-Tools
uv sync
```

### Run

```bash
uv run main.py
```

Follow the interactive prompts:
1. **Select Folder** - Choose input directory
2. **Select Operation** - Background removal, watermark removal, or image splitting
3. **Configure Settings** - Adjust parameters for your use case
4. **Process** - Let the tool handle the rest

Output files are saved in `<input-folder>/output/`

## 📚 Features In-Depth

### Background Removal

#### Ultra Backend (Maximum Quality)

**⚠️ Non-commercial use only (CC BY-NC 4.0)**

The ultimate solution for users seeking the absolute best quality:

**Features:**
- **BRIA RMBG-2.0** - Professional-grade model with superior training data
- **Trimap Refinement** - Intelligent boundary processing (preserves details)
- **Multi-space Defringing** - RGB + LAB + HSV color analysis
- **Guided Filter** - Edge-aware smoothing (better than Gaussian blur)
- **Color Filter** - Optimized for pure-color backgrounds

#### Recommended Settings

| Scenario | Strength | Color Filter |
|----------|----------|--------------|
| General photos | 0.6-0.8 | Off |
| Complex edges (hair/fur) | 0.7-0.9 | Off |
| Green screen | 0.8-0.9 | Green |
| White background (product) | 0.8-0.9 | White |
| Black background (studio) | 0.8-0.9 | Black |

#### When to Enable Color Filter?

✅ **Enable** for:
- Green screen photography/video
- Product photography (pure white/black background)
- ID photos (solid color background)

❌ **Disable** for:
- Natural scenes (complex backgrounds)
- Gradient backgrounds
- Mixed backgrounds

### Watermark Removal

Remove Google Gemini AI watermarks using reverse alpha blending algorithm.

**Features:**
- **Automatic Detection** - Auto-detect watermark size based on image dimensions
- **Manual Override** - Force 48px or 96px mode if needed
- **Strength Control** - Adjust removal intensity (0.1-1.0)
- **Precision Algorithm** - Mathematical reverse alpha blending

**Detection Rules:**
- Images > 1024×1024 pixels → 96×96 watermark
- Images ≤ 1024×1024 pixels → 48×48 watermark

### Image Splitting

Split large images into smaller tiles with intelligent layout.

**Features:**
- **Smart Grid** - Auto-calculate optimal rows × columns
- **Custom Layout** - Specify exact grid configuration
- **Quality Preservation** - Lossless PNG output
- **Batch Processing** - Process multiple images

**Use Cases:**
- Social media carousels (Instagram, Twitter)
- Print layouts and posters
- Game tile maps
- Large artwork segmentation

## 🔧 Technical Details

### Processing Pipeline (Background Removal)

```
Input Image
    ↓
Stage 1: BRIA RMBG-2.0 Segmentation
    ├─ Professional-grade alpha matte generation
    ├─ High-resolution processing (1024×1024)
    └─ Strength-based threshold adjustment
    ↓
Stage 2: Trimap Refinement (Optional)
    ├─ Identify uncertain boundary regions
    ├─ Guided filter for edge-aware smoothing
    └─ Preserve fine details (hair, fur)
    ↓
Stage 3: Multi-space Defringing
    ├─ RGB color balance analysis
    ├─ LAB color space correction
    └─ Alpha-based edge blending
    ↓
Stage 4: Color Filter (Optional)
    ├─ HSV/LAB color space detection
    ├─ Morphological mask refinement
    ├─ Edge despill (for green screens)
    └─ Alpha channel merging
    ↓
Output PNG (RGBA)
```

### Performance

- **Single Image**: 2-5 seconds (depends on resolution and hardware)
- **Memory Usage**: ~3-4GB GPU (BRIA RMBG-2.0), ~2GB CPU
- **Batch Processing**: Efficient session reuse
- **Quality**: Professional-grade, comparable to commercial services

### Hardware Requirements

| Configuration | Specs |
|---------------|-------|
| **Minimum** | 8GB RAM + CPU (slower) |
| **Recommended** | 16GB RAM + 4GB+ VRAM GPU (CUDA) |
| **Optimal** | 32GB RAM + 8GB+ VRAM GPU |

## 🆚 Comparison

| Feature | Commercial Tools | **Image-Tools** |
|---------|------------------|-----------------|
| Privacy | ❌ Upload required | ✅ 100% local |
| Customization | ❌ Limited control | ✅ Full control |
| Cost | 💰 $0.20+/image | ✅ Free (hardware only) |
| Quality | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Features | Background removal only | ✅ Multi-tool suite |
| Commercial Use | ✅ Allowed (paid) | ⚠️ Non-commercial only |

## 📦 Dependencies

All dependencies are open-source with permissive licenses:

- **transformers** (Apache 2.0) — HuggingFace transformers for BRIA RMBG-2.0
- **torch** (BSD-3) — PyTorch deep learning framework
- **opencv-python** (Apache 2.0) — Computer vision operations
- **pillow** (HPND) — Image processing utilities
- **InquirerPy** (MIT) — Interactive CLI interface
- **numpy** (BSD-3) — Numerical computing

## 🙏 Acknowledgments

This project builds upon excellent open-source work:

- **BRIA RMBG-2.0** by [BRIA AI](https://huggingface.co/briaai/RMBG-2.0) - Professional-grade model
- **BiRefNet** by [ZhengPeng7](https://github.com/ZhengPeng7/BiRefNet) - SOTA segmentation research
- **Gemini Watermark Remover** by [journey-ad](https://github.com/journey-ad/gemini-watermark-remover) - Watermark removal algorithm
- **Cloudflare** - For [evaluating background removal models](https://blog.cloudflare.com/background-removal-with-workers-ai/)

## 📄 License

**MIT License** - See [LICENSE](LICENSE) for details

### Ultra Backend License

⚠️ **Important**: The Ultra backend uses BRIA RMBG-2.0 which is licensed under **CC BY-NC 4.0** (non-commercial).

**This means:**
- ✅ Personal use is allowed and free
- ✅ Research and educational use is allowed
- ❌ Commercial use is NOT allowed without separate licensing
- For commercial use, please contact BRIA AI for licensing

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Made with ❤️ for the open-source community**
