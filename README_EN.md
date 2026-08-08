# ComfyUI-AIMDO-XPU

Intel XPU implementation of DynamicVRAM for ComfyUI (comfy-aimdo XPU replacement)

> It improves VRAM utilization by adjusting how models are loaded, reducing the chance of OOM.
> In short: OFF (full load) = "space for speed", ON (DynamicVRAM / dyvram) = "speed for space".

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

### 🆕 【New Pilot】DLL Backend (Level Zero, Hardware-level)

> [!NOTE]
> **This is a new pilot of this project**: providing a **hardware-level DynamicVRAM
> backend as a precompiled DLL**, complementing the pure-Python hijack approach below.
> It is currently distributed as a **Release pilot** — feel free to try it and give feedback.

**Origin**

- Built from the community project `xiangyuT/comfy-aimdo-xpu`
  (<https://github.com/xiangyuT/comfy-aimdo-xpu>), branch `dev/xpu-level-zero-vbar`,
  which provides an Intel **Level Zero** + oneAPI SYCL hardware-level implementation;
- This repository compiles it for Windows as `aimdo_xpu.dll` and publishes it via
  Release as a pilot, so Windows users can download and use it directly.

**Principle**

- Unlike the Python hijack approach's "LRU tensor cache simulation", the DLL backend uses
  **real Level Zero virtual address reservation (VBAR) + page-fault fault-in**, closer to the
  hardware behavior of the original NVIDIA comfy-aimdo;
- It takes effect by **replacing the `comfy_aimdo` package in `site-packages`**, not via PYTHONPATH hijack.

> [!CAUTION]
> **Note: this modifies / replaces official aimdo files**
>
> - This approach **overwrites (replaces)** the official files under
>   `site-packages/comfy_aimdo/` (`control.py`, `torch.py` and 4 other .py + `aimdo_xpu.dll`);
> - `deploy.bat` automatically backs up the original package to `comfy_aimdo.bak` for rollback;
> - **This DLL approach is mutually exclusive with the PYTHONPATH hijack approach below — only one can be active.**

**Required environment**

| # | Requirement | Notes |
|---|-------------|-------|
| 1 | Intel Arc GPU (B580 / A770 / A750, etc.) | Latest Intel graphics driver installed |
| 2 | ComfyUI-aki package (Intel ARC edition) | Bundles torch 2.13.0+xpu XPU environment |
| 3 | Intel oneAPI 2026.1 | Must match the build version of this package |
| 4 | Windows 11 / 10 (64-bit) | |

**Comparison with the hijack approach**

| Dimension | Python Hijack (default) | DLL Backend (pilot) |
|---|---|---|
| Implementation | Pure Python, zero compilation | Precompiled C/C++ (Level Zero + SYCL) |
| VBAR mechanism | LRU cache + watermark eviction (software) | Real virtual address reservation + fault-in (hardware) |
| Deployment | custom_nodes + PYTHONPATH hijack | Deploy to site-packages (replace official package) |
| Dependencies | No compilation dependency | Requires oneAPI 2026.1 (matching the build) |
| Performance focus | Lightweight, easy to share | Closer to original aimdo performance |

**How to try the DLL backend**

> [!TIP]
> - Download the latest release package (`comfy_aimdo_xpu_win_v*.zip`) from **GitHub Releases**;
> - Unzip and follow `README-DEPLOY-CN.md` / `README-DEPLOY-EN.md` inside the package;
> - ⚠️ If you previously used the hijack version of this plugin, clean it up first per the
>   "upgrade notice" at the top of the deployment doc, or the DLL will not take effect.

---

---

### Original Architecture Decision: PYTHONPATH Hijack

The original approach reimplements the above capabilities on Intel XPU with the same Python API interface, **without modifying ComfyUI's official code**.

The official comfy-aimdo is installed in `site-packages/comfy_aimdo/`. This project places a same-named `comfy_aimdo/` package under `custom_nodes/`, and uses the launch script to prepend the project path to `PYTHONPATH`, so `import comfy_aimdo` hits our project first, achieving transparent replacement.

> **Why not hijack via custom_node __init__.py?** `main.py` line 34's `import comfy.aimdo.control` runs before custom_nodes are loaded — custom_nodes `__init__.py` is already too late.

#### Replacement Strategy

| Original CUDA Mechanism | XPU Replacement |
|---|---|
| aimdo.dll VBAR virtual address page fault | torch.xpu LRU tensor cache (model_vbar.py) |
| CUDA `page fault handler` | Explicit `fault()` fault-in / fault-out |
| `cudaHostAlloc` pinned memory | `torch.empty(pin_memory=True)` |
| `VirtualAlloc + CreateFileMapping` | Python `mmap` module |
| CUDA device property query | `torch.xpu.get_device_properties()` |

#### Project Structure

```
ComfyUI-AIMDO-XPU/
├── comfy_aimdo/              ← hijacked package (dir name must match official)
│   ├── __init__.py           ← torch.cuda global shim; one-time patch on load
│   ├── control.py            ← init/init_device/get_total_vram_usage
│   ├── model_vbar.py         ← ModelVBAR + all module-level functions
│   ├── torch.py              ← aimdo_to_tensor / hostbuf_to_tensor
│   ├── host_buffer.py        ← HostBuffer (torch pin_memory instead of CUDA)
│   └── model_mmap.py         ← ModelMMAP (Python mmap instead of aimdo.dll)
├── __init__.py               ← custom_node entry; XPUAIMDOStatus node (toggle + debug)
├── README.md
└── README_EN.md
```

---

#### Download & Installation

##### Method 1: Git Clone (Recommended)

```batch
cd f:\ComfyUI-aki-v3\ComfyUI\custom_nodes
git clone https://github.com/allanmeng/ComfyUI-AIMDO-XPU.git
```

##### Method 2: Download ZIP

1. Visit the [GitHub repository](https://github.com/allanmeng/ComfyUI-AIMDO-XPU)
2. Click **Code** → **Download ZIP**
3. Extract to `ComfyUI/custom_nodes/ComfyUI-AIMDO-XPU` directory

##### Prerequisites

- Intel Arc B-Series GPU (B580 / B570 etc.)
- Intel oneAPI Base Toolkit (provides oneMKL and SYCL runtime)
- PyTorch XPU version: `pip install torch --index-url https://download.pytorch.org/whl/xpu`
- ComfyUI-aki-v3 or other ComfyUI distribution with Intel XPU support

##### Verify Installation

After launching ComfyUI, check if the logs show:

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE
[ComfyUI-AIMDO-XPU] v0.5
```

---

#### Activation

Add this project's root to `PYTHONPATH` **before** launching ComfyUI:

```batch
:: Assuming your launch script is in ComfyUI-aki-v3\ directory
set "PYTHONPATH=%~dp0ComfyUI\custom_nodes\ComfyUI-AIMDO-XPU;%PYTHONPATH%"
```

`%~dp0` automatically points to the bat file's directory, adapting to any installation path.

> Python module search order: `PYTHONPATH` > `site-packages`, so
> `import comfy_aimdo` will find our local `comfy_aimdo/` package first,
> not the official CUDA version in `site-packages`.

##### How to Disable

Comment out the `set PYTHONPATH` line to switch back to official comfy-aimdo (CUDA) for comparison testing.

##### Toggle DynamicVRAM in the Workflow

The XPU AIMDO Status node (v0.5) includes an `Enable_DynamicVRAM` switch:

<img width="45%" height="45%" alt="image" src="https://github.com/user-attachments/assets/afde7138-bdf5-4d09-88a7-d31718e81e05" />

- **ON** = VBAR staged loading (VRAM-efficient, slower inference)
- **OFF** = Full-speed full load (uses more VRAM, fastest)
- **Delete node** = auto-reverts to OFF
- **debug** checkbox enables Proxy diagnostic logs

| Mode | VRAM usage | Speed | Analogy |
|------|-----------|-------|---------|
| OFF (full load) | Very high (OOM risk) | Fastest (full speed) | All books spread on a huge desk — grab and read instantly |
| ON (dyvram) | Very low (run big images on 12GB) | Slower (depends on strategy) | Small desk, only a few books fit. Need to swap books from the shelf for each new chapter |

> Just place the node standalone in your workflow — no wiring needed.
> Toggle takes effect immediately without restarting ComfyUI. Switching OFF automatically cleans up VBAR cache and VRAM fragmentation.

---

#### Verification

##### Method 1: Check Startup Logs

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE  →  ...ComfyUI-AIMDO-XPU\comfy_aimdo\__init__.py
[ComfyUI-AIMDO-XPU] init_device(0) OK  name='Intel Arc ...'  total_vram=X.XX GB
DynamicVRAM support detected and enabled
```

##### Method 2: Workflow Node

Add the **"XPU AIMDO Status"** node to any workflow — the status report will appear in the startup logs.---

#### Differences from Original

- **No hardware-level page fault interrupts (VBAR)**: Python-layer LRU fault-in/fault-out instead; slightly lower performance but functionally equivalent
- **No CUDA dependencies whatsoever**
- `comfy.memory_management.aimdo_enabled` is set to `True`,
  `CoreModelPatcher` switches to `ModelPatcherDynamic`, identical to the original path

---

#### Known Limitations

##### ❌ NF4 / FP4 Quantization Not Supported

Intel XPU (Arc B-Series) **does not support** NF4 / FP4 quantization inference. Reason: `bitsandbytes.matmul_4bit` calls a CUDA-specific kernel; no corresponding implementation exists for the XPU backend.

**Behavior**: Using NF4/FP4 quantized models in workflows will error or fall back to non-quantized path.

**Affected models**:
- Stable Diffusion models with NF4/FP4 quantized weights (typically `-NF4`, `-FP4` suffix)
- 4-bit quantized versions exported by third-party quantization tools

**Alternatives**:
- Use **GGUF Q4_K / Q5_K / Q6_K** quantization formats (XPU supported)
- Use **GGUF-f16** / **GGUF-bf16** full-precision formats

> The `gguf qtypes` field in logs shows the actual quantization type used by the current model.
