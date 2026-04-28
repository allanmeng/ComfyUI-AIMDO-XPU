"""
ComfyUI-AIMDO-XPU  -  Custom Node entry point

This file serves two purposes:
1. Registers a lightweight diagnostic node ("XPU AIMDO Status") in ComfyUI
   so you can verify at runtime that the XPU hijack is active.
2. Prints a startup banner confirming whether comfy_aimdo was successfully
   replaced by this XPU implementation.

NOTE: The actual hijack is NOT done here. It is done by PYTHONPATH injection
before ComfyUI starts (see Stable_Start_IntelARC.bat).  By the time this
__init__.py runs, comfy_aimdo has already been imported; we just inspect it.
"""

import sys
import logging

_NODE_NAME = "ComfyUI-AIMDO-XPU"

# --------------------------------------------------------------------------
# Startup verification
# --------------------------------------------------------------------------
try:
    import comfy_aimdo
    _aimdo_file = getattr(comfy_aimdo, "__file_location__", None) or getattr(comfy_aimdo, "__file__", "unknown")

    if _aimdo_file and "ComfyUI-AIMDO-XPU" in _aimdo_file:
        print(f"[{_NODE_NAME}] ✅ XPU hijack ACTIVE  →  {_aimdo_file}", flush=True)
        logging.info(f"[{_NODE_NAME}] comfy_aimdo successfully replaced by XPU implementation")
    else:
        print(
            f"[{_NODE_NAME}] ⚠️  XPU hijack NOT active.\n"
            f"             comfy_aimdo loaded from: {_aimdo_file}\n"
            f"             → Add PYTHONPATH to Stable_Start_IntelARC.bat (see README)",
            flush=True,
        )
except Exception as e:
    print(f"[{_NODE_NAME}] ❌ Could not inspect comfy_aimdo: {e}", flush=True)


# --------------------------------------------------------------------------
# Diagnostic ComfyUI node
# --------------------------------------------------------------------------

class XPUAIMDOStatus:
    """
    A simple utility node that reports the current AIMDO backend status.
    Add it to any workflow to confirm XPU AIMDO is active.
    """

    CATEGORY = "utils/XPU"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "report"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def report(self):
        import comfy_aimdo
        import torch

        aimdo_file = getattr(comfy_aimdo, "__file_location__", None) or getattr(comfy_aimdo, "__file__", "unknown")
        xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()

        lines = [
            f"comfy_aimdo location : {aimdo_file}",
            f"XPU hijack active    : {'YES' if 'ComfyUI-AIMDO-XPU' in str(aimdo_file) else 'NO'}",
            f"torch.xpu available  : {'YES' if xpu_ok else 'NO'}",
        ]

        if xpu_ok:
            try:
                dev = torch.xpu.get_device_properties(0)
                lines.append(f"XPU device           : {dev.name}")
                lines.append(f"XPU total memory     : {dev.total_memory / 1024**3:.2f} GB")
            except Exception as e:
                lines.append(f"XPU device info      : error – {e}")

        try:
            from comfy_aimdo import model_vbar as mv
            tracked = mv._xpu_vram_tracked()
            lines.append(f"AIMDO tracked VRAM   : {tracked / 1024**2:.1f} MiB")
        except Exception:
            pass

        status = "\n".join(lines)
        print(f"[{_NODE_NAME}]\n{status}", flush=True)
        return (status,)


# --------------------------------------------------------------------------
# Node registration
# --------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "XPUAIMDOStatus": XPUAIMDOStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XPUAIMDOStatus": "XPU AIMDO Status",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
