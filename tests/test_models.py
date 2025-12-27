import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.models import ProcessConfig, is_supported_image


def test_process_config_default_output(tmp_path: Path) -> None:
    config = ProcessConfig(
        input_folder=tmp_path,
        backend_name="rembg",
        model="u2net",
        strength=0.5,
    )

    assert config.output_folder == tmp_path / "output"


def test_is_supported_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")

    text_path = tmp_path / "sample.txt"
    text_path.write_text("not an image")

    assert is_supported_image(image_path)
    assert not is_supported_image(text_path)
