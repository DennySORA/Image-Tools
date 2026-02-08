"""測試後端註冊表"""

from src.backends.registry import BackendRegistry


def main() -> None:
    """顯示所有已註冊的後端"""
    print("\n=== 已註冊的後端 ===\n")

    backends = BackendRegistry.list_backends()
    for i, backend in enumerate(backends, 1):
        print(f"{i}. {backend.name}")
        print(f"   描述: {backend.description}")
        print(f"   可用模型: {', '.join(m.name for m in backend.models)}")
        print()

    print(f"✓ 共 {len(backends)} 個後端已註冊")

    # 確認 image-splitter 已註冊
    if BackendRegistry.has_backend("image-splitter"):
        print("✓ image-splitter 後端已成功註冊！")

        # 測試建立實例
        backend = BackendRegistry.create(
            "image-splitter", model="max", strength=0.5
        )
        print(f"✓ 成功建立實例: {backend.name}")
    else:
        print("✗ image-splitter 後端未註冊")


if __name__ == "__main__":
    main()
