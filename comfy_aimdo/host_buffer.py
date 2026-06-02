"""
comfy_aimdo.host_buffer  -  XPU replacement

Mirrors the public API:
  - HostBuffer(device_id, reserved_size, hostbuf_size)
      .get_raw_address() -> int
      .as_tensor()       -> torch.Tensor   (XPU extension; used by torch.py)

ComfyUI 0.23.0+ calls HostBuffer(device_id, reserved_size, hostbuf_size).
Previously it was HostBuffer(size).
We use the last argument (hostbuf_size) as the allocation size, or allocate
nothing when it is 0 (stub placeholder created by register_load_device).

CUDA original: allocates CUDA-pinned host memory via aimdo.dll.
XPU replacement: uses torch.empty(..., pin_memory=True) when available,
  falling back to a plain CPU tensor.  The get_raw_address() method
  returns the tensor's data_ptr() so that existing callers that use the
  address as a key or pass it to ctypes still work.
"""

import logging
import torch

logger = logging.getLogger("comfy_aimdo_xpu")

# lib = None keeps sub-modules that do `from . import control; lib = control.lib`
# from crashing when they check `if lib is not None`.
from . import control
lib = control.lib   # always None in XPU build


class HostBuffer:
    """CPU-side pinned (or ordinary) buffer compatible with the CUDA API."""

    def __init__(self, device_id=0, reserved_size=0, hostbuf_size=None):
        """
        Args:
            device_id:     target XPU device index (unused; kept for compat)
            reserved_size: pre-reserved bytes  (unused; kept for compat)
            hostbuf_size:  actual allocation size in bytes.
                           When None or 0, no tensor is allocated (stub).
        """
        if hostbuf_size is None:
            hostbuf_size = reserved_size  # fallback for old-style callers

        self.device_id = int(device_id)
        self.size = int(hostbuf_size) if hostbuf_size else 0

        if self.size > 0:
            try:
                self._tensor = torch.empty(self.size, dtype=torch.uint8, pin_memory=True)
                self._pinned = True
            except Exception:
                # pin_memory may fail if CUDA is absent entirely
                self._tensor = torch.empty(self.size, dtype=torch.uint8)
                self._pinned = False
                logger.debug(
                    f"[ComfyUI-AIMDO-XPU] HostBuffer: pinned allocation failed for "
                    f"size={self.size}; using regular CPU tensor"
                )
        else:
            # Stub placeholder – no allocation needed
            self._tensor = torch.empty(0, dtype=torch.uint8)
            self._pinned = False

    def get_raw_address(self) -> int:
        """Returns the raw data pointer of the underlying storage."""
        return self._tensor.data_ptr()

    def as_tensor(self) -> torch.Tensor:
        """Returns the backing CPU tensor (XPU extension)."""
        return self._tensor

    def __del__(self):
        # Let Python/torch GC handle reclamation; nothing special needed.
        pass


# --------------------------------------------------------------------------
# Module-level functions  (ComfyUI 0.23.0+)
# --------------------------------------------------------------------------

def read_file_to_device(file_obj, offset, size, stream_ptr, dest_ptr, device_index, mark_cold=False):
    """
    Read a slice of a model file directly to XPU device memory.

    ComfyUI 0.23.0+ calls this in the TensorFileSlice hot path when the
    destination tensor is on a non-CPU device and no hostbuf is in use.

    On CUDA this is a direct cudaMemcpy into a pre-allocated VRAM buffer.
    On XPU we lack a Python API for raw device pointer writes, so the
    actual data transfer is deferred to the fallback tensor.copy_() path
    (see the monkey-patch of read_tensor_file_slice_into in __init__.py).
    """
    # Function must exist (ComfyUI imports it), but the actual work is
    # handled by the monkey-patch of read_tensor_file_slice_into that
    # returns False for the destination=None path, triggering a normal
    # tensor.to(device) copy instead.
    pass
