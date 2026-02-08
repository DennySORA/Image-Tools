# Claude Code Instructions for Image-Tools

This document contains instructions and guidelines for Claude when working on this project.

## Code Quality Standards

### 1. Code Formatting (MANDATORY)

**After ANY code changes, you MUST run:**

```bash
uv run ruff format
```

This applies to:
- ✅ Every file edit
- ✅ Every new file creation
- ✅ Before every commit
- ✅ Even for single-line changes

**Verification:**

```bash
uv run ruff format --check
```

Expected output: `X files already formatted` (no files to reformat)

### 2. Linting

Run before committing:

```bash
uv run ruff check
```

Expected output: `All checks passed!`

### 3. Type Checking

Run mypy for type safety:

```bash
uv run mypy src
```

Expected: Only known errors (cv2 type stubs) are acceptable.

## Development Workflow

### Before Every Commit:

```bash
# 1. Format code
uv run ruff format

# 2. Check linting
uv run ruff check

# 3. Type check
uv run mypy src

# 4. If all pass, commit
git add -A
git commit -m "..."
git push origin main
```

### After Dependency Changes:

```bash
# Update lockfile
uv lock

# Sync dependencies
uv sync

# Verify installation
uv run python -c "import module_name; print('OK')"
```

## Project Structure

```
Image-Tools/
├── src/
│   ├── features/
│   │   ├── background_removal/  # Only Ultra backend
│   │   ├── watermark_removal/   # Gemini watermark remover
│   │   └── image_splitting/     # Image splitter
│   ├── backends/               # Backend registry
│   ├── ui/                     # Modern CLI interface
│   └── app.py                  # Application service
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
└── docs/                       # Documentation (multi-language)
```

## Key Constraints

### Background Removal

- **ONLY Ultra backend** (BRIA RMBG-2.0)
- **License**: CC BY-NC 4.0 (non-commercial)
- No other backends should be added without discussion

### Dependencies

Keep dependencies minimal. Current core dependencies:
- transformers (BRIA RMBG-2.0)
- torch + torchvision (Deep learning)
- opencv-contrib-python (Computer vision)
- pillow (Image processing)
- InquirerPy (CLI interface)

### Code Style

- **Line length**: 88 characters (ruff default)
- **Python version**: 3.13+
- **Docstrings**: Google style
- **Type hints**: Mandatory for all functions
- **Naming**: snake_case for functions/variables, PascalCase for classes

## Commit Message Format

Use Conventional Commits:

```
<type>(<scope>): <subject>

<body>

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `style`: Code formatting
- `docs`: Documentation
- `test`: Tests
- `chore`: Maintenance

**Example:**

```
feat(ultra): add color filter for white backgrounds

Implement LAB color space detection for white background filtering.
Improves removal quality for product photography use cases.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Testing

### Run All Tests:

```bash
uv run pytest
```

### Run Quick Tests:

```bash
uv run pytest -v -k "not slow"
```

### Check Coverage:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

## Documentation Updates

When updating features, update ALL language versions:
- README.md (English)
- docs/README.zh-TW.md (Traditional Chinese)
- docs/README.zh-CN.md (Simplified Chinese)
- docs/README.ja.md (Japanese)

## Critical Reminders

1. **ALWAYS format code before committing** (`uv run ruff format`)
2. **NEVER skip linting checks** (`uv run ruff check`)
3. **Ultra backend is non-commercial** (CC BY-NC 4.0)
4. **Keep dependencies minimal** - justify new dependencies
5. **Type hints are mandatory** - no untyped functions
6. **Test before committing** - at least run quick tests

## Questions?

If you're unsure about:
- Architecture decisions → Ask the user
- Adding dependencies → Ask the user
- Removing features → Ask the user
- License implications → Ask the user

---

**Remember: Code quality is not optional. Format, lint, and type-check on every change.**
