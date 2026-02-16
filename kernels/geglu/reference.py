import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 14336)

    # In GEGLU, input x is often projected to (B, T, 2*H) then split
    # or passed as two tensors (a, b)
    # Liger-Kernel's geglu_forward takes (a, b)
    a = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    return {"a": a, "b": b}


def ref(a, b):
    # PyTorch GELU default is 'none' approximation, but Liger uses 'tanh'
    return F.gelu(a, approximate='tanh') * b
