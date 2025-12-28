import os
import drjit as dr
import mitsuba as mi

__version__ = "0.1.0"

if variant := os.getenv("MI_DEFAULT_VARIANT"):
    mi.set_variant(variant)
elif mi.variant() is None:
    if dr.has_backend(dr.JitBackend.CUDA):
        mi.set_variant("cuda_ad_mono_polarized")
    elif dr.has_backend(dr.JitBackend.LLVM):
        mi.set_variant("llvm_ad_mono_polarized")
    else:
        raise RuntimeError(
            "Mitsuba: neither the CUDA nor the LLVM backend is available."
        )
