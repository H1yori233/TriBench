from __future__ import annotations

def get_triton_hint(e: Exception) -> str | None:
    """
    Given an exception from Triton compilation or execution,
    return a human-readable hint if it matches known patterns.
    """
    msg = str(e).lower()
    cls_name = e.__class__.__name__.lower()

    # Out of bounds / memory accesses
    if "out of bounds" in msg or "memory access" in msg or "illegal" in msg:
        return "Possible out-of-bounds access. Check BLOCK_SIZE vs tensor dimensions or missing mask in tl.load/tl.store."
    
    # Shape mismatches
    if "shape mismatch" in msg or "invalid configuration" in msg or "invalid argument" in msg or "invalid value" in msg:
        return "Shape mismatch or invalid configuration. Check grid/block sizes vs tensor shapes."

    if "compilationerror" in cls_name:
        if "axis" in msg and "reduce" in msg:
            return "Invalid reduction axis."
        if "cannot be converted" in msg or "type" in msg:
            return "Dtype mismatch in Triton compiler (e.g. storing fp32 into fp16 ptr)."
        return "Triton compilation failed. Check pointer arithmetic, types, or supported operations."
    
    if "runtimeerror" in cls_name:
        if "invalid device function" in msg:
             return "Kernel not compiled for this architecture or unsupported features used."

    return None
