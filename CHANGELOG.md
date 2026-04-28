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
- `torch.compile` VAE compile not supported on XPU dynamo
