# ComfyUI-AIMDO-XPU

An early DynamicVRAM implementation for Intel Arc XPU (PYTHONPATH hijack approach)

> [!IMPORTANT]
> ## ⛔ This project is no longer maintained
>
> **comfy-aimdo for XPU was not abandoned — it was taken over by a better solution.**
>
> Intel now has a complete, self-contained XPU stack for aimdo, made of four parts:
> **`omni_xpu_kernel` + `ComfyUI-OmniXPU` + aimdo XPU provider + kitchen XPU provider**.
> After this project's last update (v1.0-alpha, 2026-08-08), it was fully switched over to that stack.
>
> 👉 **To run ComfyUI + DynamicVRAM on Intel Arc, go straight to the "Successor stack" section below.
> Do not install this plugin.**

[English](README_EN.md) | [中文](README.md)

---

## Why this project stopped

This plugin emulated the DynamicVRAM behavior of the official comfy-aimdo (NVIDIA CUDA version) on
Intel XPU using a **pure-Python hijack**: no compilation, no modification of ComfyUI source code.
Between 2026-04 and 2026-08 it was one of the few usable options for Arc A/B-series cards.

Intel later stepped into this area (the `omni` directory of `intel/llm-scaler`) and shipped a
**hardware-level Level Zero + SYCL implementation**: real virtual address reservation (VBAR) plus
page-fault fault-in, instead of a software-layer LRU cache simulation. Both quality and maintenance
effort far exceed this project, so the "software simulation" approach had served its purpose.

| | This plugin (old) | Successor stack (current) |
|---|---|---|
| Layer | Pure Python, zero compilation | C/C++ + Level Zero / SYCL |
| VBAR mechanism | LRU tensor cache + watermark eviction (simulated) | Real VA reservation + fault-in (hardware) |
| Activation | custom_nodes + `PYTHONPATH` hijack | Install provider packages + `--enable-dynamic-vram` |
| Performance | Lightweight, easy to share, slight overhead | Close to the hardware behavior of the original aimdo |
| Status | ❌ Discontinued (this repo) | ✅ Actively maintained |

> In short: **don't install the old plugin, do install the new stack.** Both target the same outcome
> (making ComfyUI's DynamicVRAM work on XPU), but the implementations are entirely different and
> **cannot be mixed**.

---

## Successor stack: four components

| Component | Role | Notes |
|---|---|---|
| **omni_xpu_kernel** | High-performance Intel XPU / SYCL kernels (attention, quantization, ...) | Extracted from `intel/llm-scaler` |
| **ComfyUI-OmniXPU** | Thin ComfyUI integration layer (custom node) | Wires the kernel into ComfyUI |
| **aimdo XPU provider** | The DynamicVRAM backend | **Replaces this plugin's hijack** |
| **kitchen XPU provider** | Inference kernel backend | XPU fork of comfy-kitchen |

### Pick the right line for your GPU (important)

Arc A-series (DG2) and B-series (BMG) follow **two different distribution lines**. Do not mix them:

| Component | A770 / DG2 (Xe-HPG) | B-Series / BMG (Battlemage) |
|---|---|---|
| **omni_xpu_kernel** | <https://github.com/Blackwood416/omni-xpu-kernel> | <https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel> |
| **ComfyUI-OmniXPU** | <https://github.com/Blackwood416/ComfyUI-OmniXPU> | <https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU> |
| **comfy-aimdo-xpu** (aimdo XPU provider) | <https://github.com/Blackwood416/comfy-aimdo-xpu> | <https://github.com/xiangyuT/comfy-aimdo-xpu> |
| **comfy-kitchen-xpu** (kitchen XPU provider) | <https://github.com/xiangyuT/comfy-kitchen-xpu> | <https://github.com/xiangyuT/comfy-kitchen-xpu> |

> [!NOTE]
> **A note on the word "official" (avoid the pitfall)**
>
> - `intel/llm-scaler` is an official Intel repository, but its `omni` directory currently targets
>   **B-Series / Battlemage** (native wheel AOT-compiled for BMG), and `0.2.0-b2` is still
>   **beta, experimental, source-build only**.
> - `Blackwood416/omni-xpu-kernel` and `Blackwood416/ComfyUI-OmniXPU` are **standalone A770 / DG2
>   forks extracted from `intel/llm-scaler`** — the upstream only covers BMG, so A-series cards need
>   this line.
> - `xiangyuT/comfy-aimdo-xpu` and `xiangyuT/comfy-kitchen-xpu` are community-maintained XPU forks.
>   Their own READMEs describe them as **experimental**, primarily validated on Ubuntu 24.04 LTS with
>   Intel Arc Pro B70; compatibility, correctness and performance on other operating systems, GPU
>   models, software environments and workloads **are not guaranteed**.
>
> Bottom line: **each component has its own scope — check which line matches your GPU before installing.**

### Prebuilt packages: user manual

📖 **"Intel GPU ComfyUI System Optimization Guide"**
<https://allanmeng.github.io/IntelGpu-ComfyUI-Collection/intel-comfyui-guide/>

Covers: background → optimization approach → installation → verification & troubleshooting →
measured data & FAQ → recovering components after a kernel upgrade.

---

## Community

We have an Intel GPU community site and a QQ group — feel free to drop by:

- Project collection: <https://allanmeng.github.io/IntelGpu-ComfyUI-Collection/>
- QQ group: **220819365** (<https://qm.qq.com/q/gls9aI3lgA>)

---

## If you are still using this plugin

This plugin **will not be adapted to new ComfyUI kernels**. Continuing to use it will break the
call conventions with the new kernel (and the new aimdo) — worst case, ComfyUI fails to start
(that already happened once with ComfyUI 0.23.0 in v0.4).

Migration steps:

1. **Remove the hijack first**: comment out or delete the `set "PYTHONPATH=..."` line in your launch
   bat. If you installed the DLL pilot, roll back `site-packages/comfy_aimdo` using the deployment
   doc's instructions.
2. **Clean up the plugin directory** (optional): delete `custom_nodes/ComfyUI-AIMDO-XPU`.
3. **Install the successor stack** for your GPU line from the table above, following the user manual.

> Rolling back is trivial: this plugin only prepends a same-named `comfy_aimdo/` package to
> `PYTHONPATH`. Remove that `set` line and `import comfy_aimdo` resolves to the official package in
> `site-packages` again.

---

## Archive: how the original approach worked (reference only — do not install)

<details>
<summary>Expand for the v1.0-alpha architecture notes</summary>

### Architecture decision: PYTHONPATH hijack

The original approach reimplemented comfy-aimdo's capabilities on Intel XPU with the same Python API,
**without modifying ComfyUI's official code**.

The official comfy-aimdo lives in `site-packages/comfy_aimdo/`. This project placed a same-named
`comfy_aimdo/` package under `custom_nodes/`, and used the launch script to prepend the project path
to `PYTHONPATH`, so `import comfy_aimdo` hit this project first — a transparent replacement.

> **Why not hijack from the custom node `__init__.py`?** `main.py`'s `import comfy.aimdo.control`
> runs before custom_nodes are loaded; by the time `__init__.py` runs, it's already too late.

#### Replacement strategy

| Original CUDA mechanism | XPU replacement |
|---|---|
| `aimdo.dll` VBAR virtual address page fault | `torch.xpu` LRU tensor cache (`model_vbar.py`) |
| CUDA page fault handler | Explicit `fault()` fault-in / fault-out |
| `cudaHostAlloc` pinned memory | `torch.empty(pin_memory=True)` |
| `VirtualAlloc + CreateFileMapping` | Python `mmap` module |
| CUDA device property query | `torch.xpu.get_device_properties()` |

#### Project structure

```
ComfyUI-AIMDO-XPU/
├── comfy_aimdo/              ← hijacked package (dir name must match official)
│   ├── __init__.py           ← torch.cuda global shim; one-time patch on load
│   ├── control.py            ← init / init_device / get_total_vram_usage
│   ├── model_vbar.py         ← ModelVBAR + all module-level functions
│   ├── torch.py              ← aimdo_to_tensor / hostbuf_to_tensor
│   ├── host_buffer.py        ← HostBuffer (torch pin_memory instead of CUDA)
│   ├── model_mmap.py         ← ModelMMAP (Python mmap instead of aimdo.dll)
│   └── vram_buffer.py        ← VRAMBuffer (new ComfyUI API)
├── __init__.py               ← custom_node entry; XPUAIMDOStatus node (toggle + debug)
├── README.md
└── README_EN.md
```

#### Activation (historical)

Before launching ComfyUI, prepend the project root to `PYTHONPATH`:

```batch
:: Assuming your launch script is in ComfyUI-aki-v3\ directory
set "PYTHONPATH=%~dp0ComfyUI\custom_nodes\ComfyUI-AIMDO-XPU;%PYTHONPATH%"
```

Python's module search order is `PYTHONPATH` > `site-packages`, so `import comfy_aimdo` finds the
local package first.

#### Verification (historical)

Startup logs should show:

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE  →  ...ComfyUI-AIMDO-XPU\comfy_aimdo\__init__.py
[ComfyUI-AIMDO-XPU] init_device(0) OK  name='Intel Arc ...'  total_vram=X.XX GB
DynamicVRAM support detected and enabled
```

#### In-workflow toggle (v0.5)

The `XPU AIMDO Status` node exposes an `Enable_DynamicVRAM` switch:

- **ON** = VBAR staged loading (VRAM-efficient, slower inference)
- **OFF** = full-speed full load (uses more VRAM, fastest)
- **Delete node** = auto-reverts to OFF; the `debug` checkbox enables Proxy diagnostic logs

Mental model: **OFF is "space for speed", ON (dyvram) is "speed for space."**
OFF spreads all books on a huge desk — instant to grab. ON has a small desk, so every new chapter
means putting books back and pulling new ones out.

#### Differences from the original

- **No hardware-level page fault interrupts (VBAR)**: Python-layer LRU fault-in/fault-out instead;
  slightly lower performance but functionally equivalent
- **No CUDA dependencies whatsoever**
- `comfy.memory_management.aimdo_enabled` is set to `True`, and `CoreModelPatcher` switches to
  `ModelPatcherDynamic` — identical to the original path

#### Known limitations

**❌ NF4 / FP4 quantization not supported**: on Intel XPU, `bitsandbytes.matmul_4bit` calls a
CUDA-specific kernel; there is no XPU implementation. It errors or falls back to the unquantized path.
Alternatives: GGUF **Q4_K / Q5_K / Q6_K**, or **GGUF-f16 / bf16**.

**❌ Fragile against kernel changes**: keeping in sync with both the official aimdo and the ComfyUI
kernel call conventions means any upstream upgrade can break the plugin — one of the reasons it was
eventually replaced by the provider model.

#### Postscript: the DLL pilot

During v1.0-alpha there was also a **DLL backend pilot**, built from the `dev/xpu-level-zero-vbar`
branch of [`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu), compiling a
Windows `aimdo_xpu.dll` that took effect by replacing `site-packages/comfy_aimdo`. That pilot has
since been superseded by the upstream provider model and is no longer maintained.

</details>

---

## Version history

Full details in [CHANGELOG.md](CHANGELOG.md).

| Version | Date | Highlights |
|---|---|---|
| v1.0-alpha | 2026-08-08 | Added DLL backend pilot; docs restructured (**final release** — project moved to the official stack afterwards) |
| v0.5 | 2026-06-02 | `Enable_DynamicVRAM` runtime toggle, `__class__` swapping, 4-step cleanup |
| v0.4 | 2026-06-02 | Fixed startup crash after ComfyUI 0.23.0; added `init_devices()` |
| v0.3 | 2026-05-17 | `model_mmap` zero-copy fix for large models (ACCESS_COPY + from_buffer) |
| v0.2 | 2026-05-04 | Added `VRAMBuffer`; API compatibility check on startup |
| v0.1 | 2026-04-28 | Initial release: XPU reimplementation of comfy-aimdo |

## License

This repository ships no separate LICENSE file. The code is kept as a **historical archive**; no
feature PRs / issues are accepted. For usage questions, please use the successor repositories and
community channels above.
