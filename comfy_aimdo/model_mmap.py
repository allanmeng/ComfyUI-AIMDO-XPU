"""
comfy_aimdo.model_mmap  -  XPU replacement

Mirrors the public API:
  - ModelMMAP(filepath)
      .get()    -> int   (pointer-like address of the memory-mapped data)
      .bounce() -> bool  (hint to invalidate cache; XPU: always returns True)

CUDA original: calls aimdo.dll's model_mmap_allocate / model_mmap_get which
  use VirtualAlloc + CreateFileMapping to map safetensors files into the CUDA
  virtual address space for zero-copy VBAR loading.

XPU replacement: uses Python's built-in mmap module to memory-map the file
  into the process address space.  get() returns ctypes.addressof() of the
  mmap buffer so that the memoryview slice in comfy/utils.py:load_safetensors()
  still works correctly (it only needs a valid CPU pointer, not a VBAR address).
"""

import logging
import mmap
import os
import ctypes

logger = logging.getLogger("comfy_aimdo_xpu")

from . import control
lib = control.lib   # always None in XPU build


class ModelMMAP:
    """Memory-maps a safetensors file using the OS mmap API."""

    def __init__(self, filepath):
        normalized = os.fspath(filepath)
        if isinstance(normalized, bytes):
            normalized = normalized.decode("utf-8")

        self._filepath = normalized
        self._file = None
        self._mm = None
        self._ptr = None

        try:
            self._file = open(normalized, "rb")
            self._mm = mmap.mmap(
                self._file.fileno(),
                0,                          # map entire file
                access=mmap.ACCESS_READ,
            )
            # Expose as a ctypes array so we can get its address.
            # from_buffer_copy always creates a writable copy, even though
            # the source mmap is read-only (checkpoint files must not be modified).
            file_size = os.path.getsize(normalized)
            self._cbuf = (ctypes.c_uint8 * file_size).from_buffer_copy(self._mm)
            self._ptr = ctypes.addressof(self._cbuf)
            logger.debug(
                f"[ComfyUI-AIMDO-XPU] ModelMMAP mapped '{normalized}' "
                f"size={file_size // (1024**2)}MiB  ptr=0x{self._ptr:x}"
            )
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"[ComfyUI-AIMDO-XPU] ModelMMAP failed for '{normalized}': {e}")

    def get(self) -> int:
        """Returns the raw memory address of the mapped data."""
        if self._ptr is None:
            raise RuntimeError("ModelMMAP is not initialised")
        return self._ptr

    def bounce(self) -> bool:
        """
        CUDA original: flushes DMA cache.
        XPU: no-op; always returns True.
        """
        return True

    def _cleanup(self):
        self._ptr = None
        self._cbuf = None
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self._cleanup()
