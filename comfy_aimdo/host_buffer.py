"""
comfy_aimdo.host_buffer  -  XPU replacement

Mirrors the public API:
  - HostBuffer(size)
      .get_raw_address() -> int
      .as_tensor()       -> torch.Tensor   (XPU extension; used by torch.py)

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

    def __init__(self, size: int):
        self.size = int(size)
        try:
            self._tensor = torch.empty(self.size, dtype=torch.uint8, pin_memory=True)
            self._pinned = True
        except Exception:
            # pin_memory may fail if CUDA is absent entirely
            self._tensor = torch.empty(self.size, dtype=torch.uint8)
            self._pinned = False
            logger.debug(
                f"[ComfyUI-AIMDO-XPU] HostBuffer: pinned allocation failed for "
                f"size={size}; using regular CPU tensor"
            )

    def get_raw_address(self) -> int:
        """Returns the raw data pointer of the underlying storage."""
        return self._tensor.data_ptr()

    def as_tensor(self) -> torch.Tensor:
        """Returns the backing CPU tensor (XPU extension)."""
        return self._tensor

    def __del__(self):
        # Let Python/torch GC handle reclamation; nothing special needed.
        pass
