"""
現代化互動式使用者介面

使用 InquirerPy 提供美觀的 CLI 互動體驗
- ESC 鍵返回上一步
- 方向鍵選擇選項
- 記住最近使用的選項
- 無確認提示，流暢操作
"""

from pathlib import Path
from typing import Any

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

    def run(self) -> ProcessConfig | None:
        """
        執行互動式設定流程（支援 ESC 返回）

        Returns:
            處理設定，若使用者取消則返回 None
        """
        self._show_welcome()

        while True:
            # 步驟 1: 選擇資料夾
            folder = self._select_folder()
            if folder is None:
                print("\n👋 ESC 已按下 - 退出程式")
                return None  # ESC 在第一步 = 退出程式

            print(f"\n✅ 已選擇資料夾: {folder}")

            # 步驟 2: 選擇操作類型
            while True:
                operation = self._select_operation()
                if operation is None:
                    print("\n🔙 ESC 已按下 - 返回資料夾選擇")
                    break  # ESC = 返回步驟 1

                print(f"\n✅ 已選擇操作: {operation}")

                # 步驟 3: 根據操作類型選擇後端
                while True:
                    backend_config = self._select_backend_for_operation(operation)
                    if backend_config is None:
                        print("\n🔙 ESC 已按下 - 返回操作選擇")
                        break  # ESC = 返回步驟 2

                    backend_name, model, strength, extra_config = backend_config
                    print(f"\n✅ 已完成設定: {backend_name} / {model} / {strength:.2f}")

                    # 建立並返回設定（直接執行，不再確認）
                    return ProcessConfig(
                        input_folder=folder,
                        backend_name=backend_name,
                        model=model,
                        strength=strength,
                        extra_config=extra_config,
                    )

    def _show_welcome(self) -> None:
        """顯示歡迎訊息"""
        print("\n" + "=" * 60)
        print("🎨  圖片處理工具  🎨".center(60))
        print("=" * 60)
        print("\n💡 提示：")
        print("   • 使用 ↑↓ 方向鍵選擇選項")
        print("   • Enter 確認選擇")
        print("   • ESC 或選擇「⬅️  返回上一步」返回上一層\n")

    def _select_folder(self) -> Path | None:
        """
        選擇資料夾

        Returns:
            資料夾路徑，若取消則返回 None
        """
        while True:  # Loop instead of recursion
            # 獲取最近使用的路徑
            recent_paths = self._history.load()
            choices: list[Choice | Separator | str] = []

            # 添加最近使用的路徑
            if recent_paths:
                choices.append(Separator("📁 最近使用"))
                choices.extend(
                    Choice(value=path, name=f"  {path.name} ({path.parent})")
                    for path in recent_paths[:5]
                    if path.exists()
                )
                choices.append(Separator())

            # 添加輸入新路徑選項
            choices.append(Choice(value="__custom__", name="📝 輸入新路徑..."))

            # 顯示選擇器
            try:
                folder = inquirer.select(  # type: ignore[attr-defined]
                    message="選擇輸入資料夾:",
                    choices=choices,
                    default=choices[1] if len(choices) > 2 else choices[0],  # noqa: PLR2004
                    mandatory=False,  # 允許 ESC
                    mandatory_message="請選擇一個資料夾",
                ).execute()
            except KeyboardInterrupt:
                return None

            # 處理 ESC (返回 None)
            if folder is None:
                return None

            # 處理自訂路徑
            if folder == "__custom__":
                try:
                    path_str = inquirer.filepath(  # type: ignore[attr-defined]
                        message="輸入資料夾路徑:",
                        default=str(Path.cwd()),
                        validate=lambda p: Path(p).exists() and Path(p).is_dir(),
                        invalid_message="路徑不存在或不是資料夾",
                        only_directories=True,
                        mandatory=False,
                    ).execute()
                except KeyboardInterrupt:
                    continue  # Ctrl+C = 返回選擇（重新循環）

                if path_str is None:
                    continue  # ESC = 返回選擇（重新循環）

                if path_str is None:
                    continue  # ESC = 返回選擇（重新循環）

                folder = Path(str(path_str))

            # 記錄到歷史
            self._history.save(folder)
            return folder  # type: ignore[no-any-return]

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
            Separator(),
            Choice(value=None, name="⬅️  返回上一步"),
        ]

        try:
            result = inquirer.select(  # type: ignore[attr-defined]
                message="選擇要執行的操作:",
                choices=choices,
                default=choices[1],  # 預設第一個操作
                mandatory=False,
            ).execute()
            # Cast to str for type checker, None is also valid
            return str(result) if result is not None else None
        except KeyboardInterrupt:
            return None

    def _select_backend_for_operation(
        self, operation: str
    ) -> tuple[str, str, float, dict[str, Any]] | None:
        """
        根據操作類型選擇對應的後端

        Args:
            operation: 操作類型

        Returns:
            (backend_name, model, strength, extra_config) 或 None
        """
        # 根據操作類型映射到對應的後端（簡化為單一後端）
        backend_map = {
            "watermark-removal": "gemini-watermark",
            "image-splitting": "image-splitter",
            "background-removal": "ultra",  # 使用極致後端（非商用，最強效果）
        }

        backend_name = backend_map.get(operation)
        if not backend_name:
            print(f"⚠️  未找到對應的後端: {operation}")
            return None

        # 直接配置後端
        return self._configure_backend(backend_name)

    def _configure_backend(  # noqa: PLR0911, C901, PLR0912
        self, backend_name: str
    ) -> tuple[str, str, float, dict[str, Any]] | None:
        """
        配置後端參數

        Args:
            backend_name: 後端名稱

        Returns:
            (backend_name, model, strength, extra_config) 或 None
        """
        backend_class = BackendRegistry.get(backend_name)
        models = backend_class.get_available_models()
        extra_config: dict[str, Any] = {}

        # 選擇模型
        if len(models) == 1:
            model = models[0]
        else:
            choices: list[Choice | Separator] = [
                Separator(f"⚙️  {backend_name} - 選擇模型"),
                *[Choice(value=m, name=f"  {m}") for m in models],
                Separator(),
                Choice(value=None, name="⬅️  返回上一步"),
            ]

            try:
                model = inquirer.select(  # type: ignore[attr-defined]
                    message="選擇模型:",
                    choices=choices,
                    default=choices[1] if len(choices) > 1 else None,
                    mandatory=False,
                ).execute()
            except KeyboardInterrupt:
                return None

            if model is None:
                return None

        # 設定強度（對於某些後端）
        if backend_name == "gemini-watermark":
            # Gemini 浮水印固定為 1.0
            strength = 1.0
        elif backend_name == "image-splitter":
            # 圖片分割使用滑桿選擇填充大小
            try:
                strength = inquirer.number(  # type: ignore[attr-defined]
                    message="設定裁切填充 (0.1-1.0, 影響透明邊距):",
                    min_allowed=0.1,
                    max_allowed=1.0,
                    default=0.5,
                    float_allowed=True,
                    mandatory=False,
                ).execute()
            except KeyboardInterrupt:
                return None

            if strength is None:
                return None
        elif backend_name in ("unified", "ultra"):
            # 統一/極致後端：強度 + 可選色彩過濾
            default_strength = 0.8 if backend_name == "ultra" else 0.7
            try:
                strength = inquirer.number(  # type: ignore[attr-defined]
                    message="設定處理強度 (0.1-1.0):",
                    min_allowed=0.1,
                    max_allowed=1.0,
                    default=default_strength,
                    float_allowed=True,
                    mandatory=False,
                ).execute()
            except KeyboardInterrupt:
                return None

            if strength is None:
                return None

            # 詢問是否啟用色彩過濾
            try:
                enable_filter = inquirer.confirm(  # type: ignore[attr-defined]
                    message="啟用純色背景過濾？（針對純黑/純白/綠幕背景）",
                    default=False,
                    mandatory=False,
                ).execute()
            except KeyboardInterrupt:
                return None

            if enable_filter is None:
                return None

            if enable_filter:
                # 選擇背景顏色
                color_choices = [
                    Choice(value="green", name="🟢 綠幕背景"),
                    Choice(value="white", name="⚪ 純白背景"),
                    Choice(value="black", name="⚫ 純黑背景"),
                    Separator(),
                    Choice(value=None, name="⬅️  返回上一步"),
                ]

                try:
                    bg_color = inquirer.select(  # type: ignore[attr-defined]
                        message="選擇背景顏色類型:",
                        choices=color_choices,
                        default=color_choices[0],
                        mandatory=False,
                    ).execute()
                except KeyboardInterrupt:
                    return None

                if bg_color is None:
                    return None

                extra_config["color_filter"] = bg_color
        else:
            # 其他背景移除使用滑桿選擇強度
            try:
                strength = inquirer.number(  # type: ignore[attr-defined]
                    message="設定處理強度 (0.1-1.0):",
                    min_allowed=0.1,
                    max_allowed=1.0,
                    default=0.5,
                    float_allowed=True,
                    mandatory=False,
                ).execute()
            except KeyboardInterrupt:
                return None

            if strength is None:
                return None

        return (backend_name, model, strength, extra_config)

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
