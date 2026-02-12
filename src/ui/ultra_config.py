"""
Ultra 背景移除後端的 UI 配置模組

提供 Ultra 後端的互動式設定流程
"""

from typing import Any

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from src.ui.history import SettingsHistory


# GREEN_SCREEN 預設配置（測試最佳參數）
DEFAULT_ULTRA_SETTINGS: dict[str, Any] = {
    "strength": 0.8,
    "color_filter": "green",
    "use_trimap_refine": False,
    "alpha_mode": "straight",
    "edge_decontamination": True,
    "resolution_mode": "1024",
}


def configure_ultra(
    backend_name: str, model: str, settings_history: SettingsHistory
) -> tuple[str, str, float, dict[str, Any]] | None:
    """
    配置 Ultra 背景移除後端的完整參數

    支援記住上一次設定，並詢問是否沿用。

    Args:
        backend_name: 後端名稱
        model: 模型名稱
        settings_history: 設定歷史記錄

    Returns:
        (backend_name, model, strength, extra_config) 或 None
    """
    saved = settings_history.load()

    # 如果有上次設定，顯示摘要並詢問操作
    if saved is not None:
        action = _ask_reuse_settings(saved)
        if action is None:
            return None  # ESC

        if action == "reuse":
            # 沿用上次設定
            strength = float(saved["strength"])
            extra_config = {k: v for k, v in saved.items() if k != "strength"}
            return (backend_name, model, strength, extra_config)

        if action == "reset":
            # 還原預設（綠幕優化）
            settings_history.save(DEFAULT_ULTRA_SETTINGS)
            strength = float(DEFAULT_ULTRA_SETTINGS["strength"])
            extra_config = {
                k: v for k, v in DEFAULT_ULTRA_SETTINGS.items() if k != "strength"
            }
            print("\n✅ 已還原為預設設定（綠幕優化）")
            return (backend_name, model, strength, extra_config)

        # action == "new"：重新設定（使用上次值作為預設）
        return _prompt_settings(backend_name, model, settings_history, defaults=saved)

    # 首次使用：使用預設值
    return _prompt_settings(
        backend_name, model, settings_history, defaults=DEFAULT_ULTRA_SETTINGS
    )


def _ask_reuse_settings(saved: dict[str, Any]) -> str | None:
    """
    顯示上次 Ultra 設定摘要，詢問是否沿用

    Args:
        saved: 上次儲存的設定

    Returns:
        "reuse"=沿用, "reset"=還原預設, "new"=重新設定, None=ESC
    """
    color = saved.get("color_filter", "none")
    trimap = "開" if saved.get("use_trimap_refine", True) else "關"
    alpha = saved.get("alpha_mode", "straight")
    decontam = "開" if saved.get("edge_decontamination", True) else "關"
    resolution = saved.get("resolution_mode", "1024")

    print("\n" + "-" * 50)
    print("📋 上次 Ultra 設定：")
    print(f"  💪 強度: {float(saved.get('strength', 0.8)):.2f}")
    print(f"  🎨 色彩過濾: {color}")
    print(f"  🔲 Trimap 精修: {trimap}")
    print(f"  📐 Alpha 模式: {alpha}")
    print(f"  🧹 邊緣去污染: {decontam}")
    print(f"  📏 解析度: {resolution}")
    print("-" * 50)

    choices = [
        Choice(value="reuse", name="✅ 使用上次的設定"),
        Choice(value="reset", name="🔄 還原預設（綠幕優化）"),
        Choice(value="new", name="⚙️  重新設定"),
        Separator(),
        Choice(value=None, name="⬅️  取消"),
    ]

    try:
        result = inquirer.select(  # type: ignore[attr-defined]
            message="選擇操作：",
            choices=choices,
            default="reuse",
            mandatory=False,
        ).execute()
    except KeyboardInterrupt:
        return None

    return result  # type: ignore[no-any-return]


def _prompt_settings(  # noqa: C901, PLR0911, PLR0912, PLR0915
    backend_name: str,
    model: str,
    settings_history: SettingsHistory,
    defaults: dict[str, Any] | None = None,
) -> tuple[str, str, float, dict[str, Any]] | None:
    """
    逐步提示使用者設定所有 Ultra 參數

    Args:
        backend_name: 後端名稱
        model: 模型名稱
        settings_history: 設定歷史記錄
        defaults: 上次設定值（作為預設值），None 表示首次

    Returns:
        (backend_name, model, strength, extra_config) 或 None
    """
    d = defaults or {}

    print("\n⚙️  Ultra 背景移除設定")
    print("-" * 40)

    # 1. 處理強度
    strength = _prompt_strength(d.get("strength", 0.8))
    if strength is None:
        return None

    # 2. 色彩過濾
    color_filter = _prompt_color_filter(d.get("color_filter", "none"))
    if color_filter is None:
        return None

    # 3. Trimap 精修
    use_trimap = _prompt_trimap(d.get("use_trimap_refine", True))
    if use_trimap is None:
        return None

    # 4. Alpha 模式
    alpha_mode = _prompt_alpha_mode(d.get("alpha_mode", "straight"))
    if alpha_mode is None:
        return None

    # 5. 邊緣去污染
    edge_decontam = _prompt_edge_decontamination(d.get("edge_decontamination", True))
    if edge_decontam is None:
        return None

    # 6. 推論解析度
    resolution_mode = _prompt_resolution(d.get("resolution_mode", "1024"))
    if resolution_mode is None:
        return None

    # 組合設定
    settings: dict[str, Any] = {
        "strength": float(strength),
        "color_filter": color_filter,
        "use_trimap_refine": use_trimap,
        "alpha_mode": alpha_mode,
        "edge_decontamination": edge_decontam,
        "resolution_mode": resolution_mode,
    }

    # 儲存設定
    settings_history.save(settings)

    extra_config = {k: v for k, v in settings.items() if k != "strength"}
    return (backend_name, model, float(strength), extra_config)


def _prompt_strength(default: float) -> float | None:
    """提示輸入處理強度"""
    try:
        strength = inquirer.number(  # type: ignore[attr-defined]
            message="處理強度 (0.1-1.0):",
            min_allowed=0.1,
            max_allowed=1.0,
            default=default,
            float_allowed=True,
            mandatory=False,
        ).execute()
    except KeyboardInterrupt:
        return None
    return strength  # type: ignore[no-any-return]


def _prompt_color_filter(default: str) -> str | None:
    """提示選擇色彩過濾"""
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
            default=default,
            mandatory=False,
        ).execute()
    except KeyboardInterrupt:
        return None
    return color_filter  # type: ignore[no-any-return]


def _prompt_trimap(default: bool) -> bool | None:
    """提示是否啟用 Trimap 精修"""
    try:
        use_trimap = inquirer.confirm(  # type: ignore[attr-defined]
            message="啟用 Trimap 邊界精修？",
            default=default,
            mandatory=False,
        ).execute()
    except KeyboardInterrupt:
        return None
    return use_trimap  # type: ignore[no-any-return]


def _prompt_alpha_mode(default: str) -> str | None:
    """提示選擇 Alpha 模式"""
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
            default=default,
            mandatory=False,
        ).execute()
    except KeyboardInterrupt:
        return None
    return alpha_mode  # type: ignore[no-any-return]


def _prompt_edge_decontamination(default: bool) -> bool | None:
    """提示是否啟用邊緣去污染"""
    try:
        edge_decontam = inquirer.confirm(  # type: ignore[attr-defined]
            message="啟用邊緣去污染？（KMeans 智能背景色估計）",
            default=default,
            mandatory=False,
        ).execute()
    except KeyboardInterrupt:
        return None
    return edge_decontam  # type: ignore[no-any-return]


def _prompt_resolution(default: str) -> str | None:
    """提示選擇推論解析度"""
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
            default=default,
            mandatory=False,
        ).execute()
    except KeyboardInterrupt:
        return None
    return resolution_mode  # type: ignore[no-any-return]
