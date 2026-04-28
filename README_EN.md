# ComfyUI-AIMDO-XPU

Intel XPU implementation of DynamicVRAM for ComfyUI (comfy-aimdo XPU replacement)

[English](README_EN.md) | [中文](README.md)

---

## Original comfy-aimdo (CUDA) Core Capabilities

comfy-aimdo (by Rattus, v0.2.12) is the core dependency of ComfyUI's DynamicVRAM feature, providing two hardware-level capabilities:

### 1. VBAR Virtual Address Page Fault Mechanism
- Relies on NVIDIA GPU's `aimdo.dll` injecting CUDA Virtual Address Base Register (VBAR)
- When model weights exceed VRAM, GPU triggers a page fault interrupt, triggering on-demand fault-in
- After fault-in, only the required portions remain in VRAM; fault-out frees space
- **Effect**: Run models larger than available VRAM without OOM, avoiding full model loading

### 2. Pinned Memory & File Mapping
- `cudaHostAlloc`: pinned memory for efficient GPU-CPU data transfer
- `VirtualAlloc + CreateFileMapping`: maps large model checkpoint files to virtual address space, avoiding full memory load
- **Effect**: Reduced memory footprint, supporting streaming load of super large models (>10GB)

> The original is entirely dependent on NVIDIA CUDA API and cannot run on Intel XPU (Arc B-Series).

---

## About ComfyUI-AIMDO-XPU

This project reimplements the above capabilities on Intel XPU with the same Python API interface, **without modifying ComfyUI's official code**.

### Architecture Decision: PYTHONPATH Hijack

The official comfy-aimdo is installed in `site-packages/comfy_aimdo/`. This project places a same-named `comfy_aimdo/` package under `custom_nodes/`, and uses the launch script to prepend the project path to `PYTHONPATH`, so `import comfy_aimdo` hits our project first, achieving transparent replacement.

> **Why not hijack via custom_node __init__.py?** `main.py` line 34's `import comfy.aimdo.control` runs before custom_nodes are loaded — custom_nodes `__init__.py` is already too late.

### Replacement Strategy

| Original CUDA Mechanism | XPU Replacement |
|---|---|
| aimdo.dll VBAR virtual address page fault | torch.xpu LRU tensor cache (model_vbar.py) |
| CUDA `page fault handler` | Explicit `fault()` fault-in / fault-out |
| `cudaHostAlloc` pinned memory | `torch.empty(pin_memory=True)` |
| `VirtualAlloc + CreateFileMapping` | Python `mmap` module |
| CUDA device property query | `torch.xpu.get_device_properties()` |

### Project Structure

```
ComfyUI-AIMDO-XPU/
├── comfy_aimdo/              ← hijacked package (dir name must match official)
│   ├── __init__.py           ← torch.cuda global shim; one-time patch on load
│   ├── control.py            ← init/init_device/get_total_vram_usage
│   ├── model_vbar.py         ← ModelVBAR + all module-level functions
│   ├── torch.py              ← aimdo_to_tensor / hostbuf_to_tensor
│   ├── host_buffer.py        ← HostBuffer (torch pin_memory instead of CUDA)
│   └── model_mmap.py         ← ModelMMAP (Python mmap instead of aimdo.dll)
├── __init__.py               ← custom_node entry; startup verification + XPUAIMDOStatus node
├── README.md
└── README_EN.md
```

---

## Milestones

| Phase | Milestone | Status | Notes |
|------|-----------|--------|-------|
| **① Foundation** | PYTHONPATH hijack channel | ✅ Done | `comfy_aimdo/` takes priority over `site-packages` |
| **② Remove CUDA** | `torch.cuda` global shim | ✅ Done | `_lazy_init`, `device()`, `get_device_properties` all intercepted |
| **③ Remove CUDA** | `is_nvidia()` DynamicVRAM guard | ✅ Done | XPU can trigger ComfyUI's DynamicVRAM init logic |
| **④ Core API** | VBAR virtual address page fault simulation | ✅ Done | `model_vbar.py` LRU cache replaces aimdo.dll VBAR |
| **⑤ Core API** | Pin Memory alignment | ✅ Done | `host_buffer.py` uses `torch.empty(pin_memory=True)` |
| **⑥ Core API** | File mapping alignment | ✅ Done | `model_mmap.py` uses Python `mmap` |
| **⑦ 3rd Party** | XPUSYSMonitor plugin compat | ✅ Done | `torch.cuda.is_available()` catches AssertionError |
| **⑧ 3rd Party** | SeedVR2 plugin compat | ✅ Done | `torch.cuda.device('cuda:N')` maps to `xpu:N` |
| **⑨ Validation** | SDXL inference | ✅ Done | DynamicVRAM active, 82s execution, no OOM |
| **⑩ 3rd Party** | Other plugins CUDA cleanup | 🔄 WIP | Fix on demand |
| **⑪ Stability** | Long-run / batch inference test | ⬜ Todo | Memory leak / resource leak verification |
| **⑫ Feature** | Other ComfyUI DynamicVRAM scenarios | ⬜ Todo | e.g. Video models, Diffusion models larger than VRAM |
| **⑬ Performance** | reserve-vram fine-tuning | ⬜ Todo | VRAM segmentation strategy for B580 11.67GB |
| **⑭ Performance** | Layer/module-level model unloading | ⬜ Todo | Finer granularity than whole-model unloading |
| **⑮ Engineering** | One-click install script | ⬜ Todo | No manual bat editing, plugin install only |
| **⑯ Upstream** | Submit PR to ComfyUI official | ⬜ Todo | Merge XPU compat into official, reduce downstream patches |

---

## Download & Installation

### Method 1: Git Clone (Recommended)

```batch
cd f:\ComfyUI-aki-v3\ComfyUI\custom_nodes
git clone https://github.com/allanmeng/ComfyUI-AIMDO-XPU.git
```

### Method 2: Download ZIP

1. Visit the [GitHub repository](https://github.com/allanmeng/ComfyUI-AIMDO-XPU)
2. Click **Code** → **Download ZIP**
3. Extract to `ComfyUI/custom_nodes/ComfyUI-AIMDO-XPU` directory

### Prerequisites

- Intel Arc B-Series GPU (B580 / B570 etc.)
- Intel oneAPI Base Toolkit (provides oneMKL and SYCL runtime)
- PyTorch XPU version: `pip install torch --index-url https://download.pytorch.org/whl/xpu`
- ComfyUI-aki-v3 or other ComfyUI distribution with Intel XPU support

### Verify Installation

After launching ComfyUI, check if the logs show:

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE
[ComfyUI-AIMDO-XPU] v0.1
```

---

## Activation

Add this project's root to `PYTHONPATH` **before** launching ComfyUI:

```batch
:: Assuming your launch script is in ComfyUI-aki-v3\ directory
set "PYTHONPATH=%~dp0ComfyUI\custom_nodes\ComfyUI-AIMDO-XPU;%PYTHONPATH%"
```

`%~dp0` automatically points to the bat file's directory, adapting to any installation path.

> Python module search order: `PYTHONPATH` > `site-packages`, so
> `import comfy_aimdo` will find our local `comfy_aimdo/` package first,
> not the official CUDA version in `site-packages`.

### Temporarily Disable

Comment out the `set PYTHONPATH` line to switch back to official comfy-aimdo (CUDA) for comparison testing.

---

## Verification

### Method 1: Check Startup Logs

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE  →  ...ComfyUI-AIMDO-XPU\comfy_aimdo\__init__.py
[ComfyUI-AIMDO-XPU] init_device(0) OK  name='Intel Arc ...'  total_vram=X.XX GB
DynamicVRAM support detected and enabled
```

### Method 2: Workflow Node

Add the **"XPU AIMDO Status"** node to any workflow — it displays a complete status report on execution.

---

## Differences from Original

- **No hardware-level page fault interrupts (VBAR)**: Python-layer LRU fault-in/fault-out instead; slightly lower performance but functionally equivalent
- **No CUDA dependencies whatsoever**
- `comfy.memory_management.aimdo_enabled` is set to `True`,
  `CoreModelPatcher` switches to `ModelPatcherDynamic`, identical to the original path

---

## Known Limitations

### ❌ NF4 / FP4 Quantization Not Supported

Intel XPU (Arc B-Series) **does not support** NF4 / FP4 quantization inference. Reason: `bitsandbytes.matmul_4bit` calls a CUDA-specific kernel; no corresponding implementation exists for the XPU backend.

**Behavior**: Using NF4/FP4 quantized models in workflows will error or fall back to non-quantized path.

**Affected models**:
- Stable Diffusion models with NF4/FP4 quantized weights (typically `-NF4`, `-FP4` suffix)
- 4-bit quantized versions exported by third-party quantization tools

**Alternatives**:
- Use **GGUF Q4_K / Q5_K / Q6_K** quantization formats (XPU supported)
- Use **GGUF-f16** / **GGUF-bf16** full-precision formats

> The `gguf qtypes` field in logs shows the actual quantization type used by the current model.
