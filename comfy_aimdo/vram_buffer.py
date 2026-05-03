"""
comfy_aimdo.vram_buffer  -  XPU replacement

Mirrors the public API added by comfy-aimdo upstream:
  - VRAMBuffer(size, device_index)
      .size()                             -> int
      .get(buffer_size, offset)           -> alloc tuple

The CUDA original reserves a virtual address range on the device and sub-allocates
from it for weight casting.  On XPU we pre-allocate a single large tensor (falling
back to CPU if the reservation exceeds available device memory) and return
sliced views via the alloc tuple protocol used by aimdo_to_tensor().
"""

import logging
import torch

logger = logging.getLogger("comfy_aimdo_xpu")


class VRAMBuffer:
    """Pre-allocated buffer on XPU for AIMDO weight casting.

    API (from ComfyUI upstream):

        buf = VRAMBuffer(reservation_size, device_index)
        buf.size()                          -> total bytes
        buf.get(slice_size, offset)         -> (self, offset, slice_size)
    """

    def __init__(self, size: int, device_id: int):
        self._size = int(size)
        self._device_id = device_id
        self.device = torch.device(f"xpu:{device_id}")

        # Pre-allocate one contiguous buffer – the CUDA version uses VMM to
        # over-commit; on XPU we try device memory and fall back to CPU when
        # the requested reservation (default 16 GiB) exceeds available VRAM.
        try:
            self._buffer = torch.empty(self._size, dtype=torch.uint8, device=self.device)
            logger.debug(
                f"[ComfyUI-AIMDO-XPU] VRAMBuffer: {self._size // (1024**3)} GiB on {self.device}"
            )
        except (RuntimeError, torch.OutOfMemoryError):
            self._buffer = torch.empty(self._size, dtype=torch.uint8, device="cpu")
            logger.debug(
                f"[ComfyUI-AIMDO-XPU] VRAMBuffer: {self._size // (1024**3)} GiB on CPU "
                f"(insufficient XPU VRAM)"
            )

        # _allocs dict so this object can act as the "vbar" half of an alloc tuple
        # consumed by comfy_aimdo.torch.aimdo_to_tensor().
        self._allocs: dict = {}

    # ------------------------------------------------------------------
    # Public API  (must match comfy-aimdo upstream)
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the total reservation size in bytes."""
        return self._size

    def get(self, buffer_size: int, offset: int):
        """Return an alloc tuple ``(self, offset, buffer_size)``.

        The result is consumed by ``comfy_aimdo.torch.aimdo_to_tensor()``
        which expects ``(vbar, key, size)`` and looks up ``vbar._allocs[key]``
        for the backing tensor.
        """
        key = offset
        if key not in self._allocs:
            self._allocs[key] = {
                "tensor": self._buffer[offset : offset + buffer_size],
                "size": buffer_size,
            }
        return (self, key, buffer_size)
