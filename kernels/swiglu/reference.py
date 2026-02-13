import torch
import torch.nn.functional as F


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    """Create inputs for SwiGLU."""
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 14336)

    a = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    
    return {"a": a, "b": b}


def ref(a: torch.Tensor, b: torch.Tensor):
    """Reference implementation of SwiGLU."""
    return F.silu(a) * b
