# 更新日志

## v0.1 (2026-04-28)

### 首次发布

- Intel XPU (Arc B-Series) 重新实现 comfy-aimdo 的 DynamicVRAM 功能
- PYTHONPATH 劫持，替换官方 CUDA 版 comfy_aimdo
- 基于 LRU 的 VBAR 模拟（model_vbar.py）
- Pin memory 通过 `torch.empty(pin_memory=True)` 替代 `cudaHostAlloc`
- 文件映射通过 Python `mmap` 替代 `VirtualAlloc + CreateFileMapping`
- `torch.cuda` shim 实现透明的 XPU 兼容
- `comfy.model_management` patch 使 XPU 启用 DynamicVRAM
- `XPU AIMDO Status` 诊断节点

### 已知限制

- NF4/FP4 量化不支持（`bitsandbytes.matmul_4bit` 无 XPU kernel）
- torch.compile VAE compile 在 XPU dynamo 上不支持

---

# Changelog

## v0.1 (2026-04-28)

### Initial Release

- Intel XPU (Arc B-Series) reimplementation of comfy-aimdo's DynamicVRAM
- PYTHONPATH hijack replacing official CUDA comfy_aimdo
- LRU-based VBAR simulation (model_vbar.py)
- Pin memory via `torch.empty(pin_memory=True)` instead of `cudaHostAlloc`
- File mapping via Python `mmap` instead of `VirtualAlloc + CreateFileMapping`
- `torch.cuda` shim for transparent XPU compatibility
- `comfy.model_management` patch to enable DynamicVRAM on XPU
- `XPU AIMDO Status` diagnostic node

### Known Limitations

- NF4/FP4 quantization not supported (`bitsandbytes.matmul_4bit` has no XPU kernel)
- torch.compile VAE compile not supported on XPU dynamo
