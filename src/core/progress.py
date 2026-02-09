"""
Rich 進度條模組

提供基於 rich 的進度條，取代基本的 sys.stdout 進度顯示
"""

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class RichProgressBar:
    """
    基於 rich 的批次處理進度條

    用法::

        with RichProgressBar(total=10) as bar:
            for path in images:
                bar.update(path.name, success=True)
    """

    def __init__(self, total: int) -> None:
        self._total = total
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._task_id = self._progress.add_task("Processing", total=total)
        self._success = 0
        self._failed = 0

    def __enter__(self) -> "RichProgressBar":
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self._progress.stop()

    def update(self, filename: str, *, success: bool) -> None:
        """
        更新進度條

        Args:
            filename: 當前處理的檔案名稱
            success: 處理是否成功
        """
        if success:
            self._success += 1
        else:
            self._failed += 1

        status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
        description = f"{filename} {status}"
        self._progress.update(self._task_id, advance=1, description=description)

    @property
    def success_count(self) -> int:
        """成功數量"""
        return self._success

    @property
    def failed_count(self) -> int:
        """失敗數量"""
        return self._failed
