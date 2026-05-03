# comfy_aimdo XPU hijack package
# This package replaces the official comfy-aimdo (CUDA) with an Intel XPU implementation.
# It is loaded via PYTHONPATH before site-packages so that all ComfyUI imports
# of comfy_aimdo land here instead of the official CUDA-only build.

from . import control
from . import model_vbar
from . import torch as torch_aimdo
from . import host_buffer
from . import model_mmap
from . import vram_buffer

__version__ = "0.2"
__file_location__ = __file__

# ------------------------------------------------------------------
# Monkey-patch ComfyUI so it treats XPU as a valid dynamic-VRAM device.
#
# Patch strategy: direct assignment to the module dict AFTER storing original
# references.  ComfyUI calls these as module-level functions; the lookups go
# through the module __dict__, so replacing them there intercepts all callers.
# ------------------------------------------------------------------
try:
    import comfy.model_management
    import torch

    # Capture originals first before overwriting
    _orig_is_nvidia = comfy.model_management.is_nvidia
    _orig_get_torch_device = comfy.model_management.get_torch_device
    _orig_is_device_xpu = comfy.model_management.is_device_xpu

    def _xpu_is_nvidia():
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        return _orig_is_nvidia()

    def _xpu_get_torch_device():
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")
        return _orig_get_torch_device()

    def _xpu_is_device_xpu(device):
        d = device if hasattr(device, 'type') else torch.device(device)
        if d.type == "xpu":
            return True
        return _orig_is_device_xpu(device)

    comfy.model_management.is_nvidia = _xpu_is_nvidia
    comfy.model_management.get_torch_device = _xpu_get_torch_device
    comfy.model_management.is_device_xpu = _xpu_is_device_xpu

    print("[ComfyUI-AIMDO-XPU] comfy.model_management patched → XPU device/type hooks active", flush=True)
except Exception as e:
    import traceback
    print(f"[ComfyUI-AIMDO-XPU] Warning: model_management patch failed: {e}", flush=True)
    traceback.print_exc()

# ------------------------------------------------------------------
# Global torch.cuda shim – redirects CUDA calls to XPU equivalents
# whenever the target device is an XPU device.
# This intercepts all torch.cuda.X() calls at the source so any future
# hard-coded torch.cuda usage is handled automatically.
# ------------------------------------------------------------------
try:
    import torch

    _XPU_STUB_PROPS = type("_XpuStubProps", (), {
        "name": "Intel(R) Arc(TM) B580 Graphics (XPU Shim)",
        "total_memory": 12 * 1024**3,
        "major": 12, "minor": 0,
        "gcnArchName": "xpu",
        "multi_processor_count": 160,
    })()

    def _is_xpu_device(device):
        if device is None:
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        d = torch.device(device) if not isinstance(device, torch.device) else device
        return d.type == "xpu"

    _orig_cuda_get_device_properties = torch.cuda.get_device_properties
    _orig_cuda_get_device_name = torch.cuda.get_device_name
    _orig_cuda_current_device = torch.cuda.current_device
    _orig_cuda_is_available = torch.cuda.is_available
    _orig_cuda_device = torch.cuda.device
    _orig_cuda_lazy_init = torch.cuda._lazy_init
    _orig_cuda_mem_get_info = torch.cuda.mem_get_info
    _orig_cuda_memory_stats = torch.cuda.memory_stats
    _orig_cuda_current_stream = torch.cuda.current_stream
    _orig_cuda_synchronize = torch.cuda.synchronize
    _orig_cuda_empty_cache = torch.cuda.empty_cache
    _orig_cuda_OutOfMemoryError = torch.cuda.OutOfMemoryError
    _orig_cuda_is_bf16_supported = torch.cuda.is_bf16_supported

    def _cuda_get_device_properties(device=None):
        if _is_xpu_device(device):
            return _XPU_STUB_PROPS
        return _orig_cuda_get_device_properties(device)

    def _cuda_get_device_name(device=None):
        if _is_xpu_device(device):
            return "Intel(R) Arc(TM) B580 Graphics (XPU)"
        return _orig_cuda_get_device_name(device)

    def _cuda_current_device():
        # Return device index 0 – this value is consumed by torch.xpu internals
        return 0

    def _cuda_is_available():
        # Return the real result but catch AssertionError when CUDA wasn't compiled in.
        # This prevents pynvml/NVML-based plugins (e.g. XPUSYSMonitor) from being
        # falsely triggered while still letting device-specific shims handle XPU paths.
        try:
            return _orig_cuda_is_available()
        except AssertionError:
            return False

    def _cuda_lazy_init():
        # Catch the "Torch not compiled with CUDA enabled" AssertionError.
        # When it fires, XPU is the actual GPU – silently return instead of crashing.
        try:
            _orig_cuda_lazy_init()
        except AssertionError:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return  # XPU is available – skip CUDA init gracefully
            raise  # Re-raise if the error is for a real reason

    def _cuda_device(device):
        # Intercepts torch.cuda.device('cuda:N') calls.
        # Plugins probing CUDA availability often call this directly.
        # When CUDA isn't compiled in, map cuda:N → xpu:N so the operation
        # still runs on the available Intel GPU instead of crashing.
        try:
            _orig_cuda_lazy_init()
            return _orig_cuda_device(device)
        except AssertionError:
            # CUDA unavailable – redirect to XPU
            d = torch.device(device) if isinstance(device, str) else device
            if d is not None and d.type == 'cuda':
                return torch.device('xpu', d.index if d.index is not None else 0)
            return torch.device('cpu')  # safe fallback

    def _cuda_mem_get_info(device=None):
        if _is_xpu_device(device):
            import torch as _t
            try:
                free = _t.xpu.memory_reserved(0)
                total = _t.xpu.get_device_properties(0).total_memory
                return (total - free, total)
            except Exception:
                return (0, 12 * 1024**3)
        return _orig_cuda_mem_get_info(device)

    def _cuda_memory_stats(device=None):
        if _is_xpu_device(device):
            return {}
        return _orig_cuda_memory_stats(device)

    def _cuda_current_stream(device=None):
        # Return a no-op stream for XPU; actual sync via xpu.synchronize()
        if _is_xpu_device(device):
            return None
        return _orig_cuda_current_stream(device)

    def _cuda_synchronize():
        try:
            torch.xpu.synchronize()
        except Exception:
            pass

    def _cuda_empty_cache():
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

    torch.cuda.get_device_properties = _cuda_get_device_properties
    torch.cuda.get_device_name = _cuda_get_device_name
    torch.cuda.current_device = _cuda_current_device
    torch.cuda.is_available = _cuda_is_available
    torch.cuda._lazy_init = _cuda_lazy_init
    torch.cuda.device = _cuda_device
    torch.cuda.mem_get_info = _cuda_mem_get_info
    torch.cuda.memory_stats = _cuda_memory_stats
    torch.cuda.current_stream = _cuda_current_stream
    torch.cuda.synchronize = _cuda_synchronize
    torch.cuda.empty_cache = _cuda_empty_cache
    torch.cuda.OutOfMemoryError = Exception  # broad catch for OOM on XPU
    torch.cuda.is_bf16_supported = lambda: True  # B580 supports BF16

    print("[ComfyUI-AIMDO-XPU] torch.cuda shim active → XPU-safe CUDA stubs installed", flush=True)
except Exception as e:
    print(f"[ComfyUI-AIMDO-XPU] Warning: torch.cuda shim failed: {e}", flush=True)
