import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 14336)
    
    x = torch.randn((B, T, H), dtype=dtype, device=device)
    return {"x": x}


def ref(x):
    return F.silu(x)
