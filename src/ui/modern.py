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
from src.data_model import ProcessConfig
from src.ui.history import PathHistory, SettingsHistory


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
        self._settings = SettingsHistory()

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
        elif backend_name == "ultra":
            return self._configure_ultra(backend_name, model)
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

        return (backend_name, model, float(strength), extra_config)

    def _configure_ultra(
        self, backend_name: str, model: str
    ) -> tuple[str, str, float, dict[str, Any]] | None:
        """
        配置 Ultra 背景移除後端的完整參數

        支援記住上一次設定，並詢問是否沿用。

        Args:
            backend_name: 後端名稱
            model: 模型名稱

        Returns:
            (backend_name, model, strength, extra_config) 或 None
        """
        saved = self._settings.load()

        # 如果有上次設定，顯示摘要並詢問是否沿用
        if saved is not None:
            reuse = self._ask_reuse_ultra_settings(saved)
            if reuse is None:
                return None  # ESC
            if reuse:
                strength = float(saved["strength"])
                extra_config = {k: v for k, v in saved.items() if k != "strength"}
                return (backend_name, model, strength, extra_config)

        # 全新設定或重新設定（使用上次值作為預設）
        return self._prompt_ultra_settings(backend_name, model, defaults=saved)

    def _ask_reuse_ultra_settings(self, saved: dict[str, Any]) -> bool | None:
        """
        顯示上次 Ultra 設定摘要，詢問是否沿用

        Args:
            saved: 上次儲存的設定

        Returns:
            True=沿用, False=重新設定, None=ESC
        """
        color = saved.get("color_filter", "none")
        trimap = "開" if saved.get("use_trimap_refine", True) else "關"
        portrait = "開" if saved.get("use_portrait_matting", False) else "關"
        alpha = saved.get("alpha_mode", "straight")
        decontam = "開" if saved.get("edge_decontamination", True) else "關"
        resolution = saved.get("resolution_mode", "1024")

        print("\n" + "-" * 50)
        print("📋 上次 Ultra 設定：")
        print(f"  💪 強度: {float(saved.get('strength', 0.8)):.2f}")
        print(f"  🎨 色彩過濾: {color}")
        print(f"  🔲 Trimap 精修: {trimap}")
        print(f"  👤 人像精修: {portrait}")
        if saved.get("use_portrait_matting", False):
            pm_str = float(saved.get("portrait_matting_strength", 0.7))
            pm_model = saved.get("portrait_matting_model", "enhanced")
            print(f"     強度: {pm_str:.2f} / 模型: {pm_model}")
        print(f"  📐 Alpha 模式: {alpha}")
        print(f"  🧹 邊緣去污染: {decontam}")
        print(f"  📏 解析度: {resolution}")
        print("-" * 50)

        try:
            result = inquirer.confirm(  # type: ignore[attr-defined]
                message="使用上次的設定？",
                default=True,
                mandatory=False,
            ).execute()
        except KeyboardInterrupt:
            return None

        return result  # type: ignore[no-any-return]

    def _prompt_ultra_settings(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        backend_name: str,
        model: str,
        defaults: dict[str, Any] | None = None,
    ) -> tuple[str, str, float, dict[str, Any]] | None:
        """
        逐步提示使用者設定所有 Ultra 參數

        Args:
            backend_name: 後端名稱
            model: 模型名稱
            defaults: 上次設定值（作為預設值），None 表示首次

        Returns:
            (backend_name, model, strength, extra_config) 或 None
        """
        d = defaults or {}

        print("\n⚙️  Ultra 背景移除設定")
        print("-" * 40)

        # 1. 處理強度
        try:
            strength = inquirer.number(  # type: ignore[attr-defined]
                message="處理強度 (0.1-1.0):",
                min_allowed=0.1,
                max_allowed=1.0,
                default=d.get("strength", 0.8),
                float_allowed=True,
                mandatory=False,
            ).execute()
        except KeyboardInterrupt:
            return None
        if strength is None:
            return None

        # 2. 色彩過濾
        saved_color = d.get("color_filter", "none")
        color_choices = [
            Choice(value="none", name="🚫 不使用"),
            Choice(value="green", name="🟢 綠幕背景"),
            Choice(value="white", name="⚪ 純白背景"),
            Choice(value="black", name="⚫ 純黑背景"),
            Separator(),
            Choice(value=None, name="⬅️  返回上一步"),
        ]
        try:
            color_filter = inquirer.select(  # type: ignore[attr-defined]
                message="純色背景過濾:",
                choices=color_choices,
                default=saved_color,
                mandatory=False,
            ).execute()
        except KeyboardInterrupt:
            return None
        if color_filter is None:
            return None

        # 3. Trimap 精修
        try:
            use_trimap = inquirer.confirm(  # type: ignore[attr-defined]
                message="啟用 Trimap 邊界精修？",
                default=d.get("use_trimap_refine", True),
                mandatory=False,
            ).execute()
        except KeyboardInterrupt:
            return None
        if use_trimap is None:
            return None

        # 4. 人像 Matting 精修
        try:
            use_portrait = inquirer.confirm(  # type: ignore[attr-defined]
                message="啟用人像 Matting 精修？（針對頭髮/邊緣）",
                default=d.get("use_portrait_matting", False),
                mandatory=False,
            ).execute()
        except KeyboardInterrupt:
            return None
        if use_portrait is None:
            return None

        portrait_strength = 0.7
        portrait_model = "enhanced"
        if use_portrait:
            # 4a. 人像精修強度
            try:
                portrait_strength = inquirer.number(  # type: ignore[attr-defined]
                    message="人像精修強度 (0.1-1.0):",
                    min_allowed=0.1,
                    max_allowed=1.0,
                    default=d.get("portrait_matting_strength", 0.7),
                    float_allowed=True,
                    mandatory=False,
                ).execute()
            except KeyboardInterrupt:
                return None
            if portrait_strength is None:
                return None

            # 4b. 人像精修模型
            pm_choices = [
                Choice(value="enhanced", name="Enhanced（推薦）"),
                Choice(value="birefnet", name="BiRefNet"),
                Separator(),
                Choice(value=None, name="⬅️  返回上一步"),
            ]
            try:
                portrait_model = inquirer.select(  # type: ignore[attr-defined]
                    message="人像精修模型:",
                    choices=pm_choices,
                    default=d.get("portrait_matting_model", "enhanced"),
                    mandatory=False,
                ).execute()
            except KeyboardInterrupt:
                return None
            if portrait_model is None:
                return None

        # 5. Alpha 模式
        alpha_choices = [
            Choice(value="straight", name="Straight（標準，適合大多數場景）"),
            Choice(value="premultiplied", name="Premultiplied（適合特定合成需求）"),
            Separator(),
            Choice(value=None, name="⬅️  返回上一步"),
        ]
        try:
            alpha_mode = inquirer.select(  # type: ignore[attr-defined]
                message="Alpha 模式:",
                choices=alpha_choices,
                default=d.get("alpha_mode", "straight"),
                mandatory=False,
            ).execute()
        except KeyboardInterrupt:
            return None
        if alpha_mode is None:
            return None

        # 6. 邊緣去污染
        try:
            edge_decontam = inquirer.confirm(  # type: ignore[attr-defined]
                message="啟用邊緣去污染？（KMeans 智能背景色估計）",
                default=d.get("edge_decontamination", True),
                mandatory=False,
            ).execute()
        except KeyboardInterrupt:
            return None
        if edge_decontam is None:
            return None

        # 7. 推論解析度
        res_choices = [
            Choice(value="1024", name="1024（平衡速度與品質）"),
            Choice(value="1536", name="1536（高品質）"),
            Choice(value="2048", name="2048（極致品質，需更多顯存）"),
            Choice(value="adaptive", name="Adaptive（自適應，根據輸入自動選擇）"),
            Separator(),
            Choice(value=None, name="⬅️  返回上一步"),
        ]
        try:
            resolution_mode = inquirer.select(  # type: ignore[attr-defined]
                message="推論解析度:",
                choices=res_choices,
                default=d.get("resolution_mode", "1024"),
                mandatory=False,
            ).execute()
        except KeyboardInterrupt:
            return None
        if resolution_mode is None:
            return None

        # 組合設定
        settings: dict[str, Any] = {
            "strength": float(strength),
            "color_filter": color_filter,
            "use_trimap_refine": use_trimap,
            "use_portrait_matting": use_portrait,
            "portrait_matting_strength": float(portrait_strength),
            "portrait_matting_model": portrait_model,
            "alpha_mode": alpha_mode,
            "edge_decontamination": edge_decontam,
            "resolution_mode": resolution_mode,
        }

        # 儲存設定
        self._settings.save(settings)

        extra_config = {k: v for k, v in settings.items() if k != "strength"}
        return (backend_name, model, float(strength), extra_config)

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
