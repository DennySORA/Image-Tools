"""
RichProgressBar 單元測試
"""

from unittest.mock import patch

import pytest
from src.core.progress import RichProgressBar


class TestRichProgressBarInit:
    """初始化測試"""

    @pytest.mark.unit
    def test_init_counters_zero(self) -> None:
        bar = RichProgressBar(total=5)
        assert bar.success_count == 0
        assert bar.failed_count == 0

    @pytest.mark.unit
    def test_init_zero_total(self) -> None:
        bar = RichProgressBar(total=0)
        assert bar.success_count == 0
        assert bar.failed_count == 0


class TestRichProgressBarContextManager:
    """Context manager 測試"""

    @pytest.mark.unit
    def test_enter_returns_self(self) -> None:
        bar = RichProgressBar(total=1)
        with bar as b:
            assert b is bar

    @pytest.mark.unit
    def test_exit_stops_progress(self) -> None:
        bar = RichProgressBar(total=1)
        with patch.object(bar._progress, "stop") as mock_stop:
            bar.__enter__()
            bar.__exit__(None, None, None)
            mock_stop.assert_called_once()

    @pytest.mark.unit
    def test_exception_propagates(self) -> None:
        with pytest.raises(ValueError, match="test error"):
            with RichProgressBar(total=1):
                raise ValueError("test error")


class TestRichProgressBarUpdate:
    """update() 方法測試"""

    @pytest.mark.unit
    def test_success_increments(self) -> None:
        with RichProgressBar(total=1) as bar:
            bar.update("img.png", success=True)
            assert bar.success_count == 1
            assert bar.failed_count == 0

    @pytest.mark.unit
    def test_failure_increments(self) -> None:
        with RichProgressBar(total=1) as bar:
            bar.update("img.png", success=False)
            assert bar.failed_count == 1
            assert bar.success_count == 0

    @pytest.mark.unit
    def test_mixed_counts(self) -> None:
        with RichProgressBar(total=5) as bar:
            bar.update("a.png", success=True)
            bar.update("b.png", success=True)
            bar.update("c.png", success=True)
            bar.update("d.png", success=False)
            bar.update("e.png", success=False)
            assert bar.success_count == 3
            assert bar.failed_count == 2

    @pytest.mark.unit
    def test_advances_progress(self) -> None:
        bar = RichProgressBar(total=2)
        with patch.object(bar._progress, "update") as mock_update:
            with bar:
                bar.update("img.png", success=True)
                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args
                assert call_kwargs.kwargs.get("advance") == 1

    @pytest.mark.unit
    def test_description_contains_filename(self) -> None:
        bar = RichProgressBar(total=1)
        with patch.object(bar._progress, "update") as mock_update:
            with bar:
                bar.update("photo.jpg", success=True)
                call_kwargs = mock_update.call_args
                assert "photo.jpg" in call_kwargs.kwargs.get("description", "")

    @pytest.mark.unit
    def test_description_ok_on_success(self) -> None:
        bar = RichProgressBar(total=1)
        with patch.object(bar._progress, "update") as mock_update:
            with bar:
                bar.update("img.png", success=True)
                desc = mock_update.call_args.kwargs.get("description", "")
                assert "OK" in desc

    @pytest.mark.unit
    def test_description_fail_on_failure(self) -> None:
        bar = RichProgressBar(total=1)
        with patch.object(bar._progress, "update") as mock_update:
            with bar:
                bar.update("img.png", success=False)
                desc = mock_update.call_args.kwargs.get("description", "")
                assert "FAIL" in desc


class TestRichProgressBarEdgeCases:
    """邊界情況測試"""

    @pytest.mark.unit
    def test_all_success(self) -> None:
        with RichProgressBar(total=5) as bar:
            for i in range(5):
                bar.update(f"img_{i}.png", success=True)
            assert bar.success_count == 5
            assert bar.failed_count == 0

    @pytest.mark.unit
    def test_all_failure(self) -> None:
        with RichProgressBar(total=5) as bar:
            for i in range(5):
                bar.update(f"img_{i}.png", success=False)
            assert bar.success_count == 0
            assert bar.failed_count == 5

    @pytest.mark.unit
    def test_unicode_filename(self) -> None:
        with RichProgressBar(total=1) as bar:
            bar.update("測試圖片.png", success=True)
            assert bar.success_count == 1
