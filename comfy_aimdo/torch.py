"""
comfy_aimdo.torch  -  XPU replacement

Mirrors the public API:
  - aimdo_to_tensor(alloc, device)       -> torch.Tensor
  - hostbuf_to_tensor(hostbuf)           -> torch.Tensor
  - get_tensor_from_raw_ptr(ptr, size, device) -> torch.Tensor

The original CUDA version uses __cuda_array_interface__ to wrap a raw VBAR
pointer into a tensor without a copy.  On XPU we instead return a view on
the stored tensor directly from the ModelVBAR registry, avoiding raw pointer
arithmetic entirely.

CUDAPluggableAllocator is stubbed out (always returns None) because:
  1. XPU has no equivalent pluggable allocator API in PyTorch yet.
  2. The original CUDA version already warns it is "experimental/unsupported".
"""

import logging
import torch

logger = logging.getLogger("comfy_aimdo_xpu")


def get_tensor_from_raw_ptr(ptr: int, size: int, device) -> torch.Tensor:
    """
    CUDA original: wraps a raw device pointer using __cuda_array_interface__.
    XPU replacement: returns an empty uint8 tensor of the correct size on the
    target device.  This is a best-effort compatibility shim; callers that
    subsequently write weights into this buffer via DMA will still work because
    the ModelVBAR LRU system manages the actual tensor objects.
    """
    try:
        return torch.empty(size, dtype=torch.uint8, device=device)
    except Exception as e:
        logger.warning(f"[ComfyUI-AIMDO-XPU] get_tensor_from_raw_ptr fallback to CPU: {e}")
        return torch.empty(size, dtype=torch.uint8, device="cpu")


def aimdo_to_tensor(alloc, device) -> torch.Tensor:
    """
    alloc = (ModelVBAR, key, size)
    Returns the tensor stored at that slot, moved to `device`.
    If not yet populated, returns an empty tensor as a placeholder.
    """
    vbar, key, size = alloc
    entry = vbar._allocs.get(key)
    if entry is not None and entry["tensor"] is not None:
        t = entry["tensor"]
        if str(t.device) != str(device):
            return t.to(device, non_blocking=True)
        return t
    # Placeholder – weight not yet assigned to this slot
    return get_tensor_from_raw_ptr(0, size, device)


def hostbuf_to_tensor(hostbuf) -> torch.Tensor:
    """
    CUDA original: wraps a pinned host buffer using ctypes + frombuffer.
    XPU replacement: the HostBuffer already stores a CPU tensor internally.
    """
    return hostbuf.as_tensor()


# --------------------------------------------------------------------------
# Stub: CUDAPluggableAllocator – not available on XPU
# --------------------------------------------------------------------------

class CUDAPluggableAllocator:
    """Stub to prevent AttributeError if any code does comfy_aimdo.torch.CUDAPluggableAllocator."""
    def __init__(self):
        logger.warning("[ComfyUI-AIMDO-XPU] CUDAPluggableAllocator is not supported on XPU; using stub")


def get_torch_allocator():
    """XPU: always returns None (same as the experimental CUDA path)."""
    return None
