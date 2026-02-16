import torch

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 64)
    N = params.get("N", 1024)
    
    x = torch.rand((B, N), dtype=dtype, device=device) + 0.1
    return {"x": x, "dim": -1}


def ref(x, dim):
    return torch.cumsum(x, dim=dim)
