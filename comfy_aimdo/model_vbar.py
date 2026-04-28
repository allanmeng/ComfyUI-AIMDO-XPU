"""
comfy_aimdo.model_vbar  -  XPU replacement

Mirrors the public API of the official CUDA model_vbar module:
  - ModelVBAR(size, device)
  - vbar_fault(alloc)
  - vbar_unpin(alloc)
  - vbar_signature_compare(a, b)
  - vbars_reset_watermark_limits()
  - vbars_analyze()

XPU Strategy (no hardware VBAR / page-fault):
  Rather than pre-reserving a virtual address range and faulting pages in on
  demand (which requires CUDA VMM or equivalent kernel support), this
  implementation tracks a set of weight tensors for each "VBAR region" and
  does explicit host↔XPU transfers driven by a watermark / LRU eviction
  policy.

  The API surface is identical to the CUDA version so ComfyUI never notices.

alloc tuple: (ModelVBAR instance, offset_key, size_bytes)
  - offset_key is a monotonically-increasing int used as a dict key
  - size_bytes is the byte count of the allocation
"""

import logging
import threading
import torch
from typing import Optional, List, Any

from . import control as _ctrl

logger = logging.getLogger("comfy_aimdo_xpu")

# --------------------------------------------------------------------------
# Global registry – all live ModelVBAR instances, for aggregate queries
# --------------------------------------------------------------------------
_registry_lock = threading.Lock()
_registry: List["ModelVBAR"] = []


def _xpu_vram_tracked() -> int:
    """Total bytes currently resident in XPU VRAM across all VBARs."""
    with _registry_lock:
        total = sum(v.loaded_size() for v in _registry)
        print(f"[ComfyUI-AIMDO-XPU] _xpu_vram_tracked() → {total / 1024**2:.1f} MiB  (registry={len(_registry)} VBARs)", flush=True)
        return total


def _xpu_vbar_dump():
    with _registry_lock:
        for v in _registry:
            logger.debug(
                f"[ComfyUI-AIMDO-XPU] VBAR device={v.device} "
                f"loaded={v.loaded_size() // (1024**2)}MiB "
                f"watermark={v.get_watermark() // (1024**2)}MiB "
                f"allocs={len(v._allocs)}"
            )


# --------------------------------------------------------------------------
# Signature helpers
# A "signature" is a bytes object (or None) representing the content-hash
# of a resident allocation.  We use a lightweight pointer+size fingerprint.
# --------------------------------------------------------------------------

def _make_signature(tensor: Optional[torch.Tensor]) -> Optional[bytes]:
    if tensor is None:
        return None
    # Use data_ptr + storage_offset + numel as a fast proxy for content identity.
    return (tensor.data_ptr(), tensor.storage_offset(), tensor.numel()).__hash__().to_bytes(8, "little", signed=False)


# --------------------------------------------------------------------------
# ModelVBAR
# --------------------------------------------------------------------------

class ModelVBAR:
    """
    Manages a pool of weight tensors for one device, providing:
      - alloc()      : register a new allocation slot
      - fault()      : ensure the allocation is resident in XPU VRAM
      - unpin()      : allow eviction of an allocation
      - loaded_size(): bytes currently resident
      - free_memory(): evict LRU allocations to free at least N bytes
    """

    def __init__(self, size: int, device: int):
        self.max_size = int(size)
        self.device = device
        self._xpu_device = torch.device(f"xpu:{device}") if _ctrl._xpu_available() else torch.device("cpu")
        self._lock = threading.Lock()
        # _allocs: key → {"tensor": Tensor|None, "size": int, "pinned": bool, "lru": int}
        self._allocs: dict = {}
        self._offset_counter = 0
        self._lru_counter = 0
        self._watermark_limit: int = size   # max bytes to keep resident

        with _registry_lock:
            _registry.append(self)

        logger.debug(f"[ComfyUI-AIMDO-XPU] ModelVBAR created device={device} max_size={size // (1024**2)}MiB")
        print(f"[ComfyUI-AIMDO-XPU] ModelVBAR registered → device={device} max_size={size // (1024**2)}MiB  registry_count={len(_registry)}", flush=True)

    # ------------------------------------------------------------------
    # alloc – returns an (self, key, size) triple  ("alloc tuple")
    # ------------------------------------------------------------------
    def alloc(self, num_bytes: int):
        num_bytes = int(num_bytes)
        with self._lock:
            # 512-byte alignment (match original)
            self._offset_counter = (self._offset_counter + 511) & ~511
            key = self._offset_counter
            if key + num_bytes > self.max_size:
                raise MemoryError("VBAR OOM (XPU)")
            self._offset_counter += num_bytes
            self._allocs[key] = {
                "tensor": None,
                "size": num_bytes,
                "pinned": False,
                "lru": 0,
            }
        return (self, key, num_bytes)

    # ------------------------------------------------------------------
    # fault – ensure alloc is resident in XPU VRAM
    # Returns a signature bytes object, or None if OOM
    # ------------------------------------------------------------------
    def fault(self, key: int, size: int) -> Optional[bytes]:
        with self._lock:
            entry = self._allocs.get(key)
            if entry is None:
                return None

            self._lru_counter += 1
            entry["lru"] = self._lru_counter

            tensor = entry["tensor"]
            if tensor is not None and tensor.device == self._xpu_device:
                # Already resident – return current signature
                return _make_signature(tensor)

            # Needs to be transferred to XPU.
            # First try to free space if we're over watermark.
            self._evict_to_watermark(size)

            if tensor is None:
                # Weight has not been set yet; return None (caller will copy)
                return None

            # Move to XPU
            try:
                entry["tensor"] = tensor.to(self._xpu_device, non_blocking=True)
                return _make_signature(entry["tensor"])
            except RuntimeError as e:
                logger.warning(f"[ComfyUI-AIMDO-XPU] fault OOM: {e}")
                return None  # signals OOM to caller

    # ------------------------------------------------------------------
    # store – called externally to hand ownership of a tensor to this VBAR
    # (maps to the original "mmap + virtual address" ownership model)
    # ------------------------------------------------------------------
    def store(self, key: int, tensor: torch.Tensor):
        with self._lock:
            entry = self._allocs.get(key)
            if entry is None:
                return
            entry["tensor"] = tensor

    # ------------------------------------------------------------------
    # unpin – allow eviction
    # ------------------------------------------------------------------
    def unpin(self, key: int, size: int):
        with self._lock:
            entry = self._allocs.get(key)
            if entry:
                entry["pinned"] = False

    # ------------------------------------------------------------------
    # Watermark / eviction
    # ------------------------------------------------------------------
    def set_watermark_limit(self, size_bytes: int):
        with self._lock:
            self._watermark_limit = int(size_bytes)

    def get_watermark(self) -> int:
        return self._watermark_limit

    def _evict_to_watermark(self, needed: int):
        """Move LRU unpinned tensors back to CPU until we have room."""
        current = self._loaded_size_unlocked()
        target = self._watermark_limit - needed
        if current <= target:
            return

        # Sort by LRU (oldest first)
        candidates = sorted(
            [(k, v) for k, v in self._allocs.items()
             if not v["pinned"] and v["tensor"] is not None
             and v["tensor"].device == self._xpu_device],
            key=lambda x: x[1]["lru"]
        )
        for key, entry in candidates:
            if current <= target:
                break
            tensor = entry["tensor"]
            entry["tensor"] = tensor.to("cpu", non_blocking=True)
            current -= entry["size"]
            logger.debug(f"[ComfyUI-AIMDO-XPU] evicted alloc key={key} size={entry['size'] // (1024**2)}MiB to CPU")

    def free_memory(self, size_bytes: int) -> int:
        """Evict unpinned tensors to free at least size_bytes; return freed bytes."""
        freed = 0
        with self._lock:
            candidates = sorted(
                [(k, v) for k, v in self._allocs.items()
                 if not v["pinned"] and v["tensor"] is not None
                 and v["tensor"].device == self._xpu_device],
                key=lambda x: x[1]["lru"]
            )
            for key, entry in candidates:
                if freed >= size_bytes:
                    break
                entry["tensor"] = entry["tensor"].to("cpu", non_blocking=True)
                freed += entry["size"]
        return freed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def _loaded_size_unlocked(self) -> int:
        return sum(
            e["size"] for e in self._allocs.values()
            if e["tensor"] is not None and e["tensor"].device == self._xpu_device
        )

    def loaded_size(self) -> int:
        with self._lock:
            return self._loaded_size_unlocked()

    def get_nr_pages(self) -> int:
        # Page size 32 MiB (matches VBAR_PAGE_SIZE in original)
        page = 32 * 1024 * 1024
        return (self.max_size + page - 1) // page

    def get_residency(self) -> List[int]:
        page = 32 * 1024 * 1024
        nr = self.get_nr_pages()
        flags = [0] * nr
        with self._lock:
            for entry in self._allocs.values():
                if entry["tensor"] is None:
                    continue
                # crude: mark entire pages covered by this alloc
                # (real implementation would need per-page tracking)
                if entry["tensor"].device == self._xpu_device:
                    flags[0] = 1  # at least mark first page resident
        return flags

    def prioritize(self):
        pass   # XPU impl: no-op; eviction is purely LRU-driven

    def deprioritize(self):
        pass

    def __del__(self):
        with _registry_lock:
            try:
                _registry.remove(self)
            except ValueError:
                pass


# --------------------------------------------------------------------------
# Module-level functions  (ComfyUI calls these directly)
# --------------------------------------------------------------------------

def vbar_fault(alloc) -> Optional[bytes]:
    """
    alloc = (ModelVBAR, key, size)
    Returns a signature (bytes) if resident, None on OOM.
    """
    if alloc is None:
        return None
    vbar, key, size = alloc
    return vbar.fault(key, size)


def vbar_unpin(alloc):
    if alloc is None:
        return
    vbar, key, size = alloc
    vbar.unpin(key, size)


def vbar_signature_compare(a, b) -> bool:
    if a is None or b is None:
        return False
    return a == b


def vbars_reset_watermark_limits():
    """Called by execution.py after each prompt run to reset watermark budgets."""
    with _registry_lock:
        for v in _registry:
            v.set_watermark_limit(v.max_size)


def vbars_analyze(verbose: bool = False) -> int:
    """
    Called by model_patcher.py to report how much VRAM AIMDO could give back.
    Returns estimated bytes that could be reclaimed.
    """
    total_reclaimable = 0
    with _registry_lock:
        for v in _registry:
            reclaimable = sum(
                e["size"] for e in v._allocs.values()
                if not e["pinned"]
                and e["tensor"] is not None
                and e["tensor"].device == torch.device(f"xpu:{v.device}")
            )
            total_reclaimable += reclaimable
            if verbose:
                logger.debug(
                    f"[ComfyUI-AIMDO-XPU] vbars_analyze device={v.device} "
                    f"reclaimable={reclaimable // (1024**2)}MiB"
                )
    return total_reclaimable
