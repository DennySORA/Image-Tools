"""
現代化互動式使用者介面

使用 InquirerPy 提供美觀的 CLI 互動體驗
- ESC 鍵返回上一步
- 方向鍵選擇選項
- 記住最近使用的選項
- 無確認提示，流暢操作
"""

from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from src.backends.registry import BackendRegistry
from src.core.models import ProcessConfig
from src.ui.history import PathHistory


class ModernUI:
    """
    現代化使用者介面

    操作流程：
    1. 選擇資料夾
    2. 選擇操作類型（水印移除/圖片分割/背景移除）
    3. 選擇後端和模型
    4. 設定參數
    5. 直接執行（無確認提示）
    """

    def __init__(self) -> None:
        """初始化 UI"""
        self._history = PathHistory()
        self._operation_order = [
            "watermark-removal",  # 水印移除
            "image-splitting",  # 圖片分割
            "background-removal",  # 背景移除
        ]

    def run(self) -> ProcessConfig | None:
        """
        執行互動式設定流程

        Returns:
            處理設定，若使用者取消則返回 None
        """
        self._show_welcome()

        # 步驟 1: 選擇資料夾
        folder = self._select_folder()
        if folder is None:
            return None

        # 步驟 2: 選擇操作類型
        operation = self._select_operation()
        if operation is None:
            # 返回步驟 1
            return self.run()

        # 步驟 3: 根據操作類型選擇後端
        backend_config = self._select_backend_for_operation(operation)
        if backend_config is None:
            # 返回步驟 2
            return self.run()

        backend_name, model, strength = backend_config

        # 建立並返回設定（直接執行，不再確認）
        return ProcessConfig(
            input_folder=folder,
            backend_name=backend_name,
            model=model,
            strength=strength,
        )

    def _show_welcome(self) -> None:
        """顯示歡迎訊息"""
        print("\n" + "=" * 60)
        print("🎨  圖片處理工具  🎨".center(60))
        print("=" * 60)
        print("\n💡 提示：使用 ↑↓ 方向鍵選擇，Enter 確認，ESC 返回\n")

    def _select_folder(self) -> Path | None:
        """
        選擇資料夾

        Returns:
            資料夾路徑，若取消則返回 None
        """
        # 獲取最近使用的路徑
        recent_paths = self._history.get_recent_input_paths()
        choices: list[Choice | Separator | str] = []

        # 添加最近使用的路徑
        if recent_paths:
            choices.append(Separator("📁 最近使用"))
            for path_str in recent_paths[:5]:  # 只顯示最近 5 個
                path = Path(path_str)
                if path.exists():
                    choices.append(
                        Choice(
                            value=path,
                            name=f"  {path.name} ({path.parent})",
                        )
                    )
            choices.append(Separator())

        # 添加輸入新路徑選項
        choices.append(Choice(value="__custom__", name="📝 輸入新路徑..."))

        # 顯示選擇器
        folder = inquirer.select(
            message="選擇輸入資料夾:",
            choices=choices,
            default=choices[1] if len(choices) > 2 else choices[0],  # type: ignore[arg-type]  # noqa: PLR2004
            vi_mode=True,  # 支援 vi 模式
        ).execute()

        # 處理取消
        if folder is None:
            return None

        # 處理自訂路徑
        if folder == "__custom__":
            path_str = inquirer.filepath(
                message="輸入資料夾路徑:",
                default=str(Path.cwd()),
                validate=lambda p: Path(p).exists() and Path(p).is_dir(),
                invalid_message="路徑不存在或不是資料夾",
                only_directories=True,
            ).execute()

            if path_str is None:
                return self._select_folder()  # 返回選擇

            folder = Path(path_str)

        # 記錄到歷史
        self._history.add_input_path(folder)
        return folder

    def _select_operation(self) -> str | None:
        """
        選擇操作類型

        Returns:
            操作類型，若取消則返回 None
        """
        choices = [
            Separator("🎯 選擇操作類型"),
            Choice(
                value="watermark-removal",
                name="🚿 水印移除 - 移除 Gemini AI 浮水印",
            ),
            Choice(
                value="image-splitting",
                name="✂️  圖片分割 - 將精靈圖分割成獨立物件",
            ),
            Choice(
                value="background-removal",
                name="🎨 背景移除 - 使用 AI 移除背景",
            ),
        ]

        return inquirer.select(
            message="選擇要執行的操作:",
            choices=choices,
            default=choices[1],  # 預設第一個操作
            vi_mode=True,
        ).execute()

    def _select_backend_for_operation(
        self, operation: str
    ) -> tuple[str, str, float] | None:
        """
        根據操作類型選擇對應的後端

        Args:
            operation: 操作類型

        Returns:
            (backend_name, model, strength) 或 None
        """
        # 根據操作類型映射到對應的後端
        backend_map = {
            "watermark-removal": ["gemini-watermark"],
            "image-splitting": ["image-splitter"],
            "background-removal": [
                "rembg",
                "transparent-background",
                "backgroundremover",
                "greenscreen",
            ],
        }

        available_backends = backend_map.get(operation, [])
        if not available_backends:
            print(f"⚠️  未找到對應的後端: {operation}")
            return None

        # 如果只有一個後端，直接使用
        if len(available_backends) == 1:
            backend_name = available_backends[0]
            return self._configure_backend(backend_name)

        # 多個後端，讓使用者選擇
        choices = [Separator(f"🔧 {operation} - 選擇後端")]
        for name in available_backends:
            backend_class = BackendRegistry.get(name)
            choices.append(
                Choice(
                    value=name,
                    name=f"  {name} - {backend_class.description}",
                )
            )

        backend_name = inquirer.select(
            message="選擇後端:",
            choices=choices,
            default=choices[1],
            vi_mode=True,
        ).execute()

        if backend_name is None:
            return None

        return self._configure_backend(backend_name)

    def _configure_backend(self, backend_name: str) -> tuple[str, str, float] | None:
        """
        配置後端參數

        Args:
            backend_name: 後端名稱

        Returns:
            (backend_name, model, strength) 或 None
        """
        backend_class = BackendRegistry.get(backend_name)
        models = backend_class.get_available_models()

        # 選擇模型
        if len(models) == 1:
            model = models[0]
        else:
            choices: list[Choice | Separator] = [
                Separator(f"⚙️  {backend_name} - 選擇模型"),
                *[Choice(value=m, name=f"  {m}") for m in models],
            ]

            model = inquirer.select(
                message="選擇模型:",
                choices=choices,
                default=choices[1] if len(choices) > 1 else None,
                vi_mode=True,
            ).execute()

            if model is None:
                return None

        # 設定強度（對於某些後端）
        if backend_name == "gemini-watermark":
            # Gemini 浮水印固定為 1.0
            strength = 1.0
        elif backend_name == "image-splitter":
            # 圖片分割使用滑桿選擇填充大小
            strength = inquirer.number(
                message="設定裁切填充 (0.1-1.0, 影響透明邊距):",
                min_allowed=0.1,
                max_allowed=1.0,
                default=0.5,
                float_allowed=True,
            ).execute()

            if strength is None:
                return None
        else:
            # 背景移除使用滑桿選擇強度
            strength = inquirer.number(
                message="設定處理強度 (0.1-1.0):",
                min_allowed=0.1,
                max_allowed=1.0,
                default=0.5,
                float_allowed=True,
            ).execute()

            if strength is None:
                return None

        return (backend_name, model, strength)

    def show_summary(self, config: ProcessConfig) -> None:
        """
        顯示處理摘要

        Args:
            config: 處理設定
        """
        print("\n" + "=" * 60)
        print("📋 處理設定摘要".center(60))
        print("=" * 60)
        print(f"\n  📁 輸入資料夾: {config.input_folder}")
        print(f"  🔧 後端: {config.backend_name}")
        print(f"  ⚙️  模型: {config.model}")
        print(f"  💪 強度: {config.strength:.2f}")
        print(f"  📂 輸出資料夾: {config.output_folder}")
        print("\n" + "=" * 60)
        print("\n⏳ 開始處理...\n")
