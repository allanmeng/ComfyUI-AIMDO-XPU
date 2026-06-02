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

__version__ = "0.5"
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

# ------------------------------------------------------------------
# Monkey-patch read_tensor_file_slice_into – ComfyUI 0.23.0 added a
# read_file_to_device() path that assumes a cudaMemcpy-like API to
# write file data to a raw device pointer (dest_ptr).  On XPU there
# is no Python API for raw pointer writes, so we intercept the calling
# function and let it fall through to the standard tensor copy.
# ------------------------------------------------------------------
try:
    import comfy.memory_management

    _orig_read_tensor_file_slice_into = comfy.memory_management.read_tensor_file_slice_into

    def _patched_read_tensor_file_slice_into(tensor, destination, stream=None, destination2=None):
        # When destination is None and destination2 is an XPU tensor,
        # the code calls host_buffer.read_file_to_device() which can't
        # write to raw XPU pointers.  Return False to fall through to
        # the standard dest2_view.copy_(tensor) fallback.
        if destination is None and destination2 is not None:
            if hasattr(destination2, "device") and hasattr(destination2.device, "type"):
                if destination2.device.type == "xpu":
                    return False
        return _orig_read_tensor_file_slice_into(tensor, destination, stream=stream, destination2=destination2)

    comfy.memory_management.read_tensor_file_slice_into = _patched_read_tensor_file_slice_into
    print("[ComfyUI-AIMDO-XPU] memory_management.read_tensor_file_slice_into patched → XPU safe fallback", flush=True)
except Exception as e:
    print(f"[ComfyUI-AIMDO-XPU] Warning: read_tensor_file_slice_into patch failed: {e}", flush=True)

# ------------------------------------------------------------------
# Per-prompt DynamicVRAM reset.
# ------------------------------------------------------------------
try:
    from execution import get_output_data as _orig_get_output_data

    _last_reset_prompt_id = None

    async def _patched_get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=None, pre_execute_cb=None, v3_data=None):
        global _last_reset_prompt_id
        if _last_reset_prompt_id != prompt_id:
            _last_reset_prompt_id = prompt_id
            from . import control as _ctrl
            # Always start each prompt in OFF mode.
            # Status node (if present) re-enables ON at the end.
            _ctrl.set_dynamic_vram(False)
        return await _orig_get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)

    import execution
    execution.get_output_data = _patched_get_output_data
    print("[ComfyUI-AIMDO-XPU] per-prompt DynamicVRAM reset active", flush=True)
except Exception as e:
    print(f"[ComfyUI-AIMDO-XPU] Warning: per-prompt reset patch failed: {e}", flush=True)

# ------------------------------------------------------------------
# Patch cache retrieval – ensure cached ModelPatcher instances have
# the correct is_dynamic() matching the current flag.
# ------------------------------------------------------------------
try:
    from comfy_execution.caching import BasicCache as _BasicCache
    _orig_get_immediate = _BasicCache._get_immediate

    async def _patched_get_immediate(self, node_id):
        value = await _orig_get_immediate(self, node_id)
        if value is not None:
            from . import control as _ctrl
            import comfy.model_patcher as _mp
            def _patch_mp(mp):
                mp.is_dynamic = lambda en=_ctrl._dynamic_vram_enabled: en
            if isinstance(value, _mp.ModelPatcher):
                _patch_mp(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, _mp.ModelPatcher):
                        _patch_mp(item)
                    elif isinstance(item, (list, tuple)):
                        for sub in item:
                            if isinstance(sub, _mp.ModelPatcher):
                                _patch_mp(sub)
        return value

    _BasicCache._get_immediate = _patched_get_immediate
    print("[ComfyUI-AIMDO-XPU] cache-retrieval is_dynamic patch active", flush=True)
except Exception as e:
    print(f"[ComfyUI-AIMDO-XPU] Warning: cache-retrieval patch failed: {e}", flush=True)

# ------------------------------------------------------------------
# CoreModelPatcher factory proxy (first-chance).
# Wraps CoreModelPatcher so that NEWLY created models (first load or
# after cache-clear) get the correct patcher type at creation time.
# CACHED models are handled by set_dynamic_vram()'s gc-level patch
# of is_dynamic() on all existing ModelPatcher instances.
# ------------------------------------------------------------------
try:
    import comfy.model_patcher
    import comfy.memory_management
    from . import control as _ctrl

    _BasePatcher = comfy.model_patcher.ModelPatcher
    _DynamicPatcher = comfy.model_patcher.ModelPatcherDynamic

    class _CoreModelPatcherProxy:
        """Factory proxy – selects ModelPatcher or ModelPatcherDynamic
        based on the current _dynamic_vram_enabled flag."""
        def __new__(cls, model=None, load_device=None, offload_device=None, size=0, weight_inplace_update=False):
            enabled = _ctrl._dynamic_vram_enabled
            if _ctrl._debug:
                print(f"[ComfyUI-AIMDO-XPU] [Proxy] {'DynamicPatcher' if enabled else 'BasePatcher'} for {model.__class__.__name__ if model else '?'} (_dynamic_vram_enabled={enabled})", flush=True)
            if enabled:
                return _DynamicPatcher(model, load_device, offload_device, size, weight_inplace_update)
            return _BasePatcher(model, load_device, offload_device, size, weight_inplace_update)

    comfy.model_patcher.CoreModelPatcher = _CoreModelPatcherProxy
    comfy.memory_management.aimdo_enabled = False
    print("[ComfyUI-AIMDO-XPU] DynamicVRAM default: OFF (proxy-based, no timing issues)", flush=True)
except Exception as e:
    print(f"[ComfyUI-AIMDO-XPU] Warning: CoreModelPatcher proxy failed: {e}", flush=True)
