import torch

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    M = params.get("M", 2048)
    N = params.get("N", 2048)
    K = params.get("K", 2048)

    A = torch.rand((M, K), dtype=dtype, device=device) + 0.1
    B = torch.rand((K, N), dtype=dtype, device=device) + 0.1
    
    return {"a": A, "b": B}


def ref(a, b):
    return torch.mm(a, b)
