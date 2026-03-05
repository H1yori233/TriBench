import torch

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 64)
    N = params.get("N", 1024)
    
    x = torch.rand((B, N), dtype=dtype, device=device).requires_grad_(True)
    grad_output = torch.randn((B, N), dtype=dtype, device=device)
    return {"x": x, "dim": -1, "grad_output": grad_output}


def ref(x, dim, grad_output):
    return torch.cumsum(x, dim=dim)

def ref_backward(x, dim, grad_output):
    x.grad = None
    y = torch.cumsum(x, dim=dim)
    y.backward(grad_output)
    return x.grad

