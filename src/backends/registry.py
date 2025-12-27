"""
後端註冊表模組

實現工廠模式和策略模式，遵循開放封閉原則 (OCP)
新增後端只需使用 @register 裝飾器，無需修改此模組
"""

from collections.abc import Callable
from typing import Any, cast

from src.core.interfaces import BackendProtocol, BaseBackend
from src.core.models import BackendInfo, ModelInfo


class BackendRegistry:
    """
    後端註冊表

    使用裝飾器模式註冊後端，實現開放封閉原則
    """

    _backends: dict[str, type[BaseBackend]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BaseBackend]], type[BaseBackend]]:
        """
        註冊後端的裝飾器

        Args:
            name: 後端名稱

        Returns:
            裝飾器函數

        Example:
            @BackendRegistry.register("my-backend")
            class MyBackend(BaseBackend):
                ...
        """

        def decorator(backend_class: type[BaseBackend]) -> type[BaseBackend]:
            cls._backends[name] = backend_class
            return backend_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseBackend]:
        """
        取得後端類別

        Args:
            name: 後端名稱

        Returns:
            後端類別

        Raises:
            KeyError: 當後端不存在時
        """
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise KeyError(f"後端 '{name}' 不存在，可用後端: {available}")
        return cls._backends[name]

    @classmethod
    def create(
        cls, name: str, model: str, strength: float, **kwargs: Any
    ) -> BackendProtocol:
        """
        建立後端實例

        工廠方法：根據名稱建立對應的後端實例

        Args:
            name: 後端名稱
            model: 模型名稱
            strength: 去背強度
            **kwargs: 其他參數

        Returns:
            後端實例
        """
        backend_class = cls.get(name)
        constructor = cast(Callable[..., BackendProtocol], backend_class)

        # 根據不同後端調整參數名稱
        if name == "transparent-background":
            return constructor(mode=model, strength=strength, **kwargs)
        return constructor(model=model, strength=strength, **kwargs)

    @classmethod
    def list_backends(cls) -> list[BackendInfo]:
        """
        列出所有已註冊的後端

        Returns:
            後端資訊列表
        """
        result = []
        for name, backend_class in cls._backends.items():
            models = tuple(
                ModelInfo(name=m, description="")
                for m in backend_class.get_available_models()
            )
            result.append(
                BackendInfo(
                    name=name,
                    description=backend_class.description,
                    models=models,
                )
            )
        return result

    @classmethod
    def get_backend_names(cls) -> list[str]:
        """取得所有後端名稱"""
        return list(cls._backends.keys())

    @classmethod
    def has_backend(cls, name: str) -> bool:
        """檢查後端是否存在"""
        return name in cls._backends
