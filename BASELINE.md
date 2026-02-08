# Baseline Metrics - Before Refactor

**Date**: 2026-02-08
**Commit**: Stage 0 - Baseline establishment

## Code Metrics

- **Total Lines of Code**: ~3,516 lines (source only)
- **Total Source Files**: 22 files in `src/`
- **Test Files**: 6 files in `tests/`
- **Test Count**: 64 tests (44 passed, 20 skipped due to network/model access)

## File Size Distribution

### Files Exceeding Limits
- `src/utils/splitter.py`: **574 lines** ❌ (exceeds 500-line hard cap)
- `src/backends/ultra.py`: **510 lines** ❌ (exceeds 500-line hard cap)

### Files Near Limit
- `src/backends/unified.py`: 403 lines ⚠️
- `src/backends/gemini_watermark.py`: 393 lines ⚠️
- `src/ui/modern.py`: 370 lines ⚠️

### Other Files
- `src/postprocess/green_screen.py`: 236 lines ✓
- `src/utils/union_find.py`: 134 lines ✓
- `src/backends/image_splitter.py`: 139 lines ✓
- `src/backends/registry.py`: 123 lines ✓
- `src/core/processor.py`: 134 lines ✓
- `src/core/models.py`: 135 lines ✓
- `src/core/interfaces.py`: 108 lines ✓
- `src/utils/geometry.py`: 98 lines ✓
- `src/ui/history.py`: 78 lines ✓

## Test Results

```
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
collected 64 items

44 passed, 20 skipped in 189.55s (0:03:09)
```

**Skipped Tests**: 20 tests skipped due to HuggingFace model access restrictions (BRIA RMBG-2.0 is gated)

## Quality Tools

- **mypy**: Installed ✓
- **ruff**: Installed ✓
- **pytest**: Installed ✓
- **Validation Script**: `scripts/validate.sh` created ✓

## Current Architecture

```
src/
├── backends/        # Backend implementations (registry pattern)
├── core/           # Core interfaces and models
├── postprocess/    # Post-processing utilities
├── ui/             # User interface (CLI)
└── utils/          # Utility modules
```

## Known Issues

1. **File Length**: 2 files exceed 500-line hard cap, 3 files exceed 300-line target
2. **Code Duplication**: `ColorFilter` and `ColorFilterConfig` duplicated in `ultra.py` and `unified.py`
3. **Missing Pydantic**: Using `dataclass` instead of Pydantic models
4. **No BaseSettings**: No centralized configuration management
5. **Mixed Concerns**: `utils/` folder mixes different responsibilities
6. **No Feature Structure**: Flat backend organization

## Refactor Goals

1. Add Pydantic models and BaseSettings
2. Extract common code and eliminate duplication
3. Restructure into feature-based architecture
4. Break down large files (<= 300 lines target, 500 hard cap)
5. Enhance SOLID principles and dependency injection
6. Enforce strict typing (Python 3.13 conventions)
7. Pass all quality gates (mypy, ruff, tests)
