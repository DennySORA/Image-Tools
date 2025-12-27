[繁體中文](docs/README.zh-TW.md) | [简体中文](docs/README.zh-CN.md) | [日本語](docs/README.ja.md)

# Remove Background

A user-friendly tool for removing image backgrounds with multiple methods and a guided flow. It is designed for batch processing and for handling both everyday photos and green-screen shots.

## Installation
- Make sure Python 3.13 and `uv` are available.
- Clone the repository and install dependencies.

```bash
git clone <repository-url>
cd Remove-Background
uv sync
```

## How to Use
- Start the interactive flow and follow the prompts.
- Outputs are saved as transparent PNGs in an `output/` folder under the selected directory.

```bash
uv run main.py
```

## Usage Options
- Guided interactive flow: select a folder, pick a removal method, adjust strength, and confirm before processing.
- Green-screen workflow: optimized for solid green backdrops to clean edges and reduce color spill.
- General photo workflow: balanced removal for portraits, products, and mixed backgrounds.
- Speed vs quality choices: pick a mode that fits your time and visual expectations.

## Key Features
- Multiple removal methods with clear descriptions.
- Batch processing with progress feedback and a results summary.
- Adjustable strength to control how aggressive the removal is.
- Consistent transparent-background results for downstream use.

## External Dependencies (Third-Party)
- **backgroundremover** — External dependency. Background removal option.
- **rembg** — External dependency. Background removal option with multiple choices.
- **transparent-background** — External dependency. Background removal option.
- **onnxruntime** — External dependency. Runtime support for the above options.
- **pillow** — External dependency. Image reading and writing utilities.
- **moviepy** — External dependency. Media utility library listed in dependencies.
