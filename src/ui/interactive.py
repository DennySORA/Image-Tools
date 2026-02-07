"""
交互式使用者介面模組

提供交互式的設定流程，遵循單一職責原則
負責使用者互動，不處理業務邏輯
"""

from pathlib import Path

from src.backends.registry import BackendRegistry
from src.core.models import SUPPORTED_EXTENSIONS, ProcessConfig, ProcessResult

from .console import Console
from .history import PathHistory


# 強度範圍常數
MIN_STRENGTH: float = 0.1
MAX_STRENGTH: float = 1.0
DEFAULT_STRENGTH: float = 0.5


class InteractiveUI:
    """
    交互式使用者介面

    負責引導使用者完成設定流程，支援返回上一步
    """

    def __init__(self) -> None:
        """初始化 UI"""
        self._console = Console()
        self._history = PathHistory()

    def run(self) -> ProcessConfig | None:
        """
        執行交互式設定流程

        Returns:
            處理設定，若使用者取消則返回 None
        """
        self._show_welcome()

        # 使用狀態機模式支援返回上一步
        folder: Path | None = None
        backend_name: str | None = None
        model: str | None = None
        strength: float | None = None

        # 步驟 1: 選擇資料夾
        while folder is None:
            folder = self._select_folder()
            if folder is None:
                return None  # 使用者在第一步取消

        # 步驟 2: 選擇後端
        while backend_name is None:
            backend_name = self._select_backend()
            if backend_name is None:
                # 返回步驟 1
                folder = None
                continue

            # 步驟 3: 選擇模型
            model = self._select_model(backend_name)
            if model is None:
                # 返回步驟 2
                backend_name = None
                continue

            # 步驟 4: 設定強度 (gemini-watermark 固定為 1.0)
            if backend_name == "gemini-watermark":
                strength = 1.0
            else:
                strength = self._select_strength()
                if strength is None:
                    # 返回步驟 3
                    model = None
                    backend_name = None
                    continue

            # 建立設定
            config = ProcessConfig(
                input_folder=folder,
                backend_name=backend_name,
                model=model,
                strength=strength,
            )

            # 確認設定
            confirmed = self._confirm_settings(config)
            if confirmed is None:
                # 返回步驟 4
                strength = None
                model = None
                backend_name = None
                continue
            if not confirmed:
                return None

            return config

        return None

    def _show_welcome(self) -> None:
        """顯示歡迎畫面"""
        self._console.clear()
        self._console.print_header("圖片背景移除工具 (Interactive Mode)")

        self._console.write_line(
            f"支援的圖片格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        self._console.write_line("輸出格式: PNG (保留透明通道)")
        self._console.write_line("\n提示: 任何步驟輸入 'b' 可返回上一步")

    def _select_folder(self) -> Path | None:
        """
        選擇資料夾

        Returns:
            資料夾路徑，若無效或取消則返回 None
        """
        self._console.print_section("【步驟 1/4】選擇圖片資料夾")

        history = self._history.load()

        if history:
            folder = self._select_from_history(history)
        else:
            folder = self._input_new_folder()

        if folder is None:
            return None

        self._history.save(folder)
        return folder

    def _select_from_history(self, history: list[Path]) -> Path | None:
        """
        從歷史記錄中選擇路徑

        Args:
            history: 歷史路徑列表

        Returns:
            選擇的資料夾路徑，若無效或返回則為 None
        """
        options = [str(p) for p in history] + ["輸入新路徑"]
        choice = self._console.get_choice(
            "最近使用的路徑:", options, default=1, allow_back=False
        )

        if choice is None or choice == len(options):
            return self._input_new_folder()

        return self._validate_folder(history[choice - 1])

    def _input_new_folder(self) -> Path | None:
        """
        手動輸入新資料夾路徑

        Returns:
            資料夾路徑，若無效則返回 None
        """
        while True:
            folder_path = self._console.get_input(
                "請輸入資料夾路徑 (b 取消)"
            )

            if folder_path.lower() in ("b", "back", "返回"):
                return None

            if not folder_path:
                self._console.write_line("路徑不能為空")
                continue

            folder = Path(folder_path).expanduser().resolve()
            result = self._validate_folder(folder)
            if result is not None:
                return result

    def _validate_folder(self, folder: Path) -> Path | None:
        """
        驗證資料夾路徑

        Args:
            folder: 資料夾路徑

        Returns:
            驗證通過的路徑，驗證失敗返回 None
        """
        if not folder.exists():
            self._console.write_line(f"錯誤: 資料夾不存在 - {folder}")
            return None

        if not folder.is_dir():
            self._console.write_line(f"錯誤: 路徑不是資料夾 - {folder}")
            return None

        # 掃描圖片
        image_count = sum(
            1
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if image_count == 0:
            self._console.write_line("錯誤: 資料夾中沒有找到支援的圖片檔案")
            self._console.write_line(
                f"支援的格式: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
            return None

        self._console.write_line(f"\n找到 {image_count} 張圖片")
        return folder

    def _select_backend(self) -> str | None:
        """
        選擇後端

        Returns:
            後端名稱，若返回上一步則為 None
        """
        self._console.print_section("【步驟 2/4】選擇背景移除方案")

        backends = BackendRegistry.list_backends()
        options = [f"{b.name}: {b.description}" for b in backends]

        choice = self._console.get_choice(
            "請選擇背景移除方案:", options, default=1, allow_back=True
        )

        if choice is None:
            return None

        return backends[choice - 1].name

    def _select_model(self, backend_name: str) -> str | None:
        """
        選擇模型

        Args:
            backend_name: 後端名稱

        Returns:
            模型名稱，若返回上一步則為 None
        """
        self._console.print_section("【步驟 3/4】選擇模型")

        backend_class = BackendRegistry.get(backend_name)
        models = backend_class.get_available_models()

        # 根據不同後端顯示不同的選項描述
        if backend_name == "rembg":
            options = self._get_rembg_model_options(models)
        elif backend_name == "transparent-background":
            options = self._get_transparent_bg_mode_options(models)
        elif backend_name == "greenscreen":
            options = self._get_greenscreen_mode_options(models)
        elif backend_name == "gemini-watermark":
            options = self._get_gemini_watermark_mode_options(models)
        else:
            options = self._get_backgroundremover_model_options(models)

        choice = self._console.get_choice(
            "請選擇模型:", options, default=1, allow_back=True
        )

        if choice is None:
            return None

        return models[choice - 1]

    def _get_rembg_model_options(self, models: list[str]) -> list[str]:
        """取得 Rembg 模型選項"""
        descriptions = {
            "birefnet-general": "BiRefNet 通用 - 效果最好 (推薦)",
            "birefnet-general-lite": "BiRefNet 輕量版 - 速度較快",
            "birefnet-portrait": "BiRefNet 人像 - 人像專用",
            "birefnet-massive": "BiRefNet 大型 - 大型資料集訓練",
            "isnet-general-use": "ISNet 通用",
            "isnet-anime": "ISNet 動漫 - 動漫角色專用",
            "u2net": "U2Net 經典",
            "u2netp": "U2Net 輕量版",
            "u2net_human_seg": "U2Net 人體分割",
            "silueta": "Silueta - U2Net 壓縮版",
        }
        return [f"{m}: {descriptions.get(m, m)}" for m in models]

    def _get_transparent_bg_mode_options(self, models: list[str]) -> list[str]:
        """取得 Transparent Background 模式選項"""
        descriptions = {
            "base": "預設模式 - 平衡速度與品質 (推薦)",
            "fast": "快速模式 - 速度較快但品質稍低",
            "base-nightly": "Nightly 版本 - 可能有最新改進",
        }
        return [f"{m}: {descriptions.get(m, m)}" for m in models]

    def _get_backgroundremover_model_options(self, models: list[str]) -> list[str]:
        """取得 BackgroundRemover 模型選項"""
        descriptions = {
            "u2net": "U2Net 通用 (推薦)",
            "u2net_human_seg": "U2Net 人像 - 精度最高",
            "u2netp": "U2Net 輕量版 - 速度較快",
        }
        return [f"{m}: {descriptions.get(m, m)}" for m in models]

    def _get_greenscreen_mode_options(self, models: list[str]) -> list[str]:
        """取得 GreenScreen 模式選項"""
        descriptions = {
            "hybrid": "混合模式 - 色度鍵+AI+Despill，效果最好 (推薦)",
            "chroma-only": "純色度鍵 - 速度最快，適合純色綠幕",
            "ai-enhanced": "AI增強 - 色度鍵+AI，保留原始色彩",
        }
        return [f"{m}: {descriptions.get(m, m)}" for m in models]

    def _get_gemini_watermark_mode_options(
        self, models: list[str]
    ) -> list[str]:
        """取得 Gemini 浮水印移除模式選項"""
        descriptions = {
            "auto": "自動偵測 - 依圖片尺寸自動選擇浮水印大小 (推薦)",
            "48px": "48×48 模式 - 強制使用小尺寸浮水印模式",
            "96px": "96×96 模式 - 強制使用大尺寸浮水印模式",
        }
        return [f"{m}: {descriptions.get(m, m)}" for m in models]

    def _select_strength(self) -> float | None:
        """
        選擇去背強度

        Returns:
            強度值 (0.1-1.0)，若返回上一步則為 None
        """
        self._console.print_section("【步驟 4/4】設定去背強度")

        self._console.write_line("強度說明:")
        self._console.write_line("  - 較低 (0.3-0.5): 保守去背，保留更多邊緣細節")
        self._console.write_line("  - 中等 (0.5-0.7): 平衡模式")
        self._console.write_line(
            "  - 較高 (0.7-1.0): 積極去背，邊緣更乾淨但可能損失細節"
        )

        while True:
            value = input(
                f"\n去背強度 [{MIN_STRENGTH}-{MAX_STRENGTH}] "
                f"(預設: {DEFAULT_STRENGTH}, b 返回): "
            ).strip()
            if value.lower() in ("b", "back", "返回"):
                return None
            if not value:
                return DEFAULT_STRENGTH
            try:
                num = float(value)
                if MIN_STRENGTH <= num <= MAX_STRENGTH:
                    return num
                self._console.write_line(
                    f"請輸入 {MIN_STRENGTH} 到 {MAX_STRENGTH} 之間的數值"
                )
            except ValueError:
                self._console.write_line("請輸入有效的數字")

    def _confirm_settings(self, config: ProcessConfig) -> bool | None:
        """
        確認設定

        Args:
            config: 處理設定

        Returns:
            是否確認，None 表示返回上一步
        """
        self._console.write_line("\n" + "=" * 60)
        self._console.write_line("確認設定")
        self._console.write_line("=" * 60)
        self._console.write_line(f"  資料夾: {config.input_folder}")
        self._console.write_line(f"  後端:   {config.backend_name}")
        self._console.write_line(f"  模型:   {config.model}")
        self._console.write_line(f"  強度:   {config.strength}")
        output_folder = config.output_folder or (config.input_folder / "output")
        self._console.write_line(f"  輸出:   {output_folder}")
        self._console.write_line("")

        while True:
            response = input("確定開始處理? [Y/n/b]: ").strip().lower()
            if not response or response in ("y", "yes", "是"):
                return True
            if response in ("n", "no", "否"):
                return False
            if response in ("b", "back", "返回"):
                return None
            self._console.write_line("請輸入 y (確定), n (取消), 或 b (返回)")

    def show_result(self, result: ProcessResult) -> None:
        """
        顯示處理結果

        Args:
            result: 處理結果
        """
        self._console.print_separator()
        self._console.write_line(
            f"\n處理完成: {result.success}/{result.total} 張圖片成功"
        )

        if result.failed > 0:
            self._console.write_line(f"失敗: {result.failed} 張")

        self._console.write_line(f"輸出位置: {result.output_folder}")

    def show_cancelled(self) -> None:
        """顯示取消訊息"""
        self._console.write_line("\n已取消")

    def ask_continue(self) -> bool:
        """
        詢問是否繼續處理

        Returns:
            True 繼續，False 退出
        """
        return self._console.confirm("\n是否繼續處理其他圖片?", default=True)
