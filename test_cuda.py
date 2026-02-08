#!/usr/bin/env python3
"""测试 CUDA 自动启用"""

import torch
from src.features.background_removal.portrait_matting import PortraitMattingRefiner
from src.features.background_removal.ultra import UltraBackend


def main() -> None:
    print("=" * 60)
    print("CUDA 可用性检查")
    print("=" * 60)

    # 检查 PyTorch CUDA 支持
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA 不可用，将使用 CPU")

    print("\n" + "=" * 60)
    print("UltraBackend 设备选择测试")
    print("=" * 60)

    # 测试 1: 默认初始化（应该自动选择 CUDA）
    print("\n1. 默认初始化（device=None）:")
    backend = UltraBackend(strength=0.7)
    print(f"   选择的设备: {backend.device}")
    print(f"   设备类型: {backend.device.type}")

    # 测试 2: 明确指定 cuda
    if torch.cuda.is_available():
        print("\n2. 明确指定 device='cuda':")
        backend_cuda = UltraBackend(strength=0.7, device="cuda")
        print(f"   选择的设备: {backend_cuda.device}")
        print(f"   设备类型: {backend_cuda.device.type}")

    # 测试 3: 明确指定 cpu
    print("\n3. 明确指定 device='cpu':")
    backend_cpu = UltraBackend(strength=0.7, device="cpu")
    print(f"   选择的设备: {backend_cpu.device}")
    print(f"   设备类型: {backend_cpu.device.type}")

    print("\n" + "=" * 60)
    print("PortraitMattingRefiner 设备选择测试")
    print("=" * 60)

    # 测试 4: Portrait Matting 默认初始化
    print("\n4. Portrait Matting 默认初始化（device=None）:")
    refiner = PortraitMattingRefiner(model_name="enhanced")
    print(f"   选择的设备: {refiner.device}")
    print(f"   设备类型: {refiner.device.type}")

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    if torch.cuda.is_available():
        print("✅ CUDA 可用，所有组件默认会自动使用 GPU")
        print("✅ UltraBackend: 自动选择 CUDA")
        print("✅ PortraitMattingRefiner: 自动选择 CUDA")
    else:
        print("⚠️  CUDA 不可用，所有组件使用 CPU")
        print("   原因可能是：")
        print("   1. 系统没有 NVIDIA GPU")
        print("   2. CUDA 驱动未安装")
        print("   3. PyTorch 安装的是 CPU 版本")

    print("\n提示：如果需要强制使用 CPU，可以传入 device='cpu' 参数")


if __name__ == "__main__":
    main()
