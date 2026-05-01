"""
comfy_aimdo.control  -  XPU replacement
Mirrors the public API of the official CUDA comfy-aimdo control module so that
ComfyUI's main.py can call init() / init_device() / get_total_vram_usage() etc.
without modification.

Key differences from the CUDA original:
  - No aimdo.dll / aimdo.so loaded; all memory introspection goes through
    torch.xpu and Intel Level Zero Python bindings where available.
  - init_device() returns True only when an Intel XPU device is detected and
    torch.xpu is functional.
  - get_total_vram_usage() tracks VRAM occupied by ModelVBAR objects managed
    by this XPU implementation.
"""

import logging
import torch

# --------------------------------------------------------------------------
# Global state
# lib is kept as None to signal "no native binary loaded".
# Sub-modules (host_buffer, model_vbar, etc.) check `lib is not None` before
# attempting to use ctypes bindings; since we have none we guard differently.
# --------------------------------------------------------------------------
lib = None          # kept for API compatibility; always None in XPU build

_initialised = False
_device_id: int = -1
_log_level = logging.INFO

logger = logging.getLogger("comfy_aimdo_xpu")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _xpu_total_memory(device_id: int = 0) -> int:
    """Return total XPU VRAM in bytes, or 0 on failure."""
    try:
        if _xpu_available():
            props = torch.xpu.get_device_properties(device_id)
            return props.total_memory
    except Exception:
        pass
    return 0


# --------------------------------------------------------------------------
# API compat check – verify all public interfaces exist
# --------------------------------------------------------------------------

_api_check_ran = False

def _check_api_compat():
    """
    Scan the full comfy_aimdo package and print a per-function compat report.
    Runs once at first init(); subsequent calls are no-ops.
    """
    global _api_check_ran
    if _api_check_ran:
        return
    _api_check_ran = True

    import comfy_aimdo as _pkg

    # (display_name, checker) pairs
    _checks = [
        # ── control module ──
        ("control.init",                       lambda: callable(_pkg.control.init)),
        ("control.init_device",                lambda: callable(_pkg.control.init_device)),
        ("control.deinit",                     lambda: callable(_pkg.control.deinit)),
        ("control.get_total_vram_usage",       lambda: callable(_pkg.control.get_total_vram_usage)),
        ("control.analyze",                    lambda: callable(_pkg.control.analyze)),
        ("control.lib",                        lambda: hasattr(_pkg.control, "lib")),
        ("control.set_log_* (×8)",             lambda: all(
            hasattr(_pkg.control, f"set_log_{lvl}")
            for lvl in ["none", "critical", "error", "warning", "info", "debug", "verbose", "vverbose"]
        )),
        # ── model_vbar module ──
        ("model_vbar.ModelVBAR",               lambda: callable(_pkg.model_vbar.ModelVBAR)),
        ("model_vbar.vbar_fault",              lambda: callable(_pkg.model_vbar.vbar_fault)),
        ("model_vbar.vbar_unpin",              lambda: callable(_pkg.model_vbar.vbar_unpin)),
        ("model_vbar.vbar_signature_compare",  lambda: callable(_pkg.model_vbar.vbar_signature_compare)),
        ("model_vbar.vbars_reset_watermark_limits", lambda: callable(_pkg.model_vbar.vbars_reset_watermark_limits)),
        ("model_vbar.vbars_analyze",           lambda: callable(_pkg.model_vbar.vbars_analyze)),
        # ── torch_aimdo module (imported as "torch" alias in comfy_aimdo/__init__.py) ──
        ("torch.aimdo_to_tensor",              lambda: callable(_pkg.torch_aimdo.aimdo_to_tensor)),
        ("torch.hostbuf_to_tensor",            lambda: callable(_pkg.torch_aimdo.hostbuf_to_tensor)),
        ("torch.get_tensor_from_raw_ptr",      lambda: callable(_pkg.torch_aimdo.get_tensor_from_raw_ptr)),
        ("torch.get_torch_allocator",          lambda: callable(_pkg.torch_aimdo.get_torch_allocator)),
        ("torch.CUDAPluggableAllocator",       lambda: callable(_pkg.torch_aimdo.CUDAPluggableAllocator)),
        # ── host_buffer module ──
        ("host_buffer.HostBuffer",             lambda: callable(_pkg.host_buffer.HostBuffer)),
        # ── model_mmap module ──
        ("model_mmap.ModelMMAP",               lambda: callable(_pkg.model_mmap.ModelMMAP)),
    ]

    total = len(_checks)
    failed = []

    for name, check in _checks:
        try:
            if not check():
                failed.append(name)
        except Exception:
            failed.append(name)

    passed = total - len(failed)

    # ── Print report ──
    print(f"[ComfyUI-AIMDO-XPU] API compat check ─────────────────", flush=True)
    for name, _ in _checks:
        icon = "✅" if name not in failed else "❌"
        print(f"  {icon}  {name}", flush=True)
    if failed:
        print(f"  ❌  MISSING: {', '.join(failed)}", flush=True)
    print(f"───────────────────────────────── {passed}/{total} aligned {'✅' if passed == total else '⚠️'}", flush=True)


# --------------------------------------------------------------------------
# Public API  (must match comfy-aimdo 0.2.x exactly)
# --------------------------------------------------------------------------

def init() -> bool:
    """
    Called by main.py before init_device().
    Prepares the XPU backend; returns True if XPU is usable.
    """
    global _initialised
    if _initialised:
        return True

    if not _xpu_available():
        print("[ComfyUI-AIMDO-XPU] torch.xpu not available – XPU AIMDO inactive", flush=True)
        return False

    _initialised = True

    # Run API compat check once at startup
    _check_api_compat()

    print(
        f"[ComfyUI-AIMDO-XPU] control.init() OK  "
        f"(torch.xpu device count={torch.xpu.device_count()})",
        flush=True,
    )
    return True


def init_device(device_id: int) -> bool:
    """
    Called by main.py after init().
    Returns True to signal that DynamicVRAM should be enabled.
    """
    global _device_id

    if not _initialised:
        print("[ComfyUI-AIMDO-XPU] init_device called before init()", flush=True)
        return False

    if not _xpu_available():
        return False

    print(f"[ComfyUI-AIMDO-XPU] init_device({device_id}) called...", flush=True)

    try:
        print(f"[ComfyUI-AIMDO-XPU] calling torch.xpu.get_device_properties({device_id})...", flush=True)
        props = torch.xpu.get_device_properties(device_id)
        print(f"[ComfyUI-AIMDO-XPU] got props: {props}", flush=True)
        _device_id = device_id
        total_gb = props.total_memory / (1024 ** 3)

        # ── NF4/FP4 compat: patch ModelPatcherDynamic to skip VBAR for Params4bit ─
        # NF4/FP4 quantized models (bitsandbytes Params4bit) store quant_state
        # as a Python attribute that points to the underlying tensor's data_ptr.
        # VBAR's fault/store moves the tensor between XPU↔CPU, which changes
        # data_ptr, but Params4bit.quant_state is NOT updated → matmul_4bit fails.
        #
        # Fix: patch _load_list() so that any module whose weight is a Params4bit
        # is excluded from VBAR allocation. It will fall through to the direct
        # cast path (force_load=True branch) and stay resident on XPU.
        try:
            _patch_model_patcher_dynamic_for_params4bit()
        except Exception as e:
            import traceback
            print(f"[ComfyUI-AIMDO-XPU] WARNING: Params4bit patch failed: {e}", flush=True)
            traceback.print_exc()

        return True

    except Exception as e:
        import traceback
        print(f"[ComfyUI-AIMDO-XPU] init_device({device_id}) failed: {e}", flush=True)
        traceback.print_exc()
        return False


def deinit():
    global _initialised, _device_id
    _initialised = False
    _device_id = -1
    print("[ComfyUI-AIMDO-XPU] deinit()", flush=True)


def get_total_vram_usage() -> int:
    """
    Returns bytes of XPU VRAM currently tracked by the XPU AIMDO allocator.
    Used by comfy/windows.py to compute available system RAM.
    """
    from . import model_vbar as _mv
    return _mv._xpu_vram_tracked()


def analyze():
    """Debug dump; called when --verbose DEBUG is set."""
    from . import model_vbar as _mv
    _mv._xpu_vbar_dump()


# --------------------------------------------------------------------------
# Log-level API  (called by main.py depending on --verbose flag)
# --------------------------------------------------------------------------

def _set_level(level: int):
    global _log_level
    _log_level = level
    logging.getLogger("comfy_aimdo_xpu").setLevel(level)


def set_log_none():     _set_level(logging.CRITICAL + 10)
def set_log_critical(): _set_level(logging.CRITICAL)
def set_log_error():    _set_level(logging.ERROR)
def set_log_warning():  _set_level(logging.WARNING)
def set_log_info():     _set_level(logging.INFO)
def set_log_debug():    _set_level(logging.DEBUG)
def set_log_verbose():  _set_level(logging.DEBUG - 1)
def set_log_vverbose(): _set_level(logging.DEBUG - 2)


# --------------------------------------------------------------------------
# NF4/FP4 compat – skip VBAR for Params4bit modules in ModelPatcherDynamic
# --------------------------------------------------------------------------

def _is_params4bit(weight) -> bool:
    """Detect bitsandbytes Params4bit / FP4 objects by their quant_state attr."""
    if weight is None:
        return False
    return hasattr(weight, "_quantized_stats") or (
        hasattr(weight, "quant_state") and weight.quant_state is not None
    )


def _patch_model_patcher_dynamic_for_params4bit():
    """Monkey-patch ModelPatcherDynamic._load_list to skip Params4bit modules.

    Modules whose weight is a Params4bit will have comfy_cast_weights=True
    but will NOT get a VBAR slot. Instead they fall through to the
    force_load=True branch and stay resident on XPU – safe from VBAR's
    CPU↔XPU eviction which would corrupt quant_state.
    """
    import comfy.model_patcher
    orig = comfy.model_patcher.ModelPatcherDynamic._load_list

    def _patched_load_list(self, for_dynamic=False, default_device=None):
        loading = orig(self, for_dynamic=for_dynamic, default_device=default_device)
        if not for_dynamic:
            return loading

        # Filter out entries whose weight is Params4bit (NF4/FP4).
        # These will take the direct-cast path (force_load) in ModelPatcherDynamic.load()
        # and won't go through VBAR eviction, preserving quant_state integrity.
        filtered = []
        skipped = 0
        for entry in loading:
            # entry: (sort_key..., module_mem, module_name, module, params_dict)
            *rest, module_mem, n, m, params = entry
            try:
                import comfy.model_patcher as mp
                weight, _, _ = mp.get_key_weight(self.model, f"{n}.weight")
                if _is_params4bit(weight):
                    skipped += 1
                    continue
            except Exception:
                pass
            filtered.append(entry)

        if skipped:
            print(f"[ComfyUI-AIMDO-XPU] Params4bit compat: skipped {skipped} module(s) from VBAR", flush=True)
        return filtered

    comfy.model_patcher.ModelPatcherDynamic._load_list = _patched_load_list
    print("[ComfyUI-AIMDO-XPU] Params4bit compat patch applied – DynamicVRAM active for all models", flush=True)

