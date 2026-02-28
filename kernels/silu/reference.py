import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 14336)
    
    x = torch.randn((B, T, H), dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn((B, T, H), dtype=dtype, device=device)
    return {"x": x, "grad_output": grad_output}


def ref(x, grad_output):
    return F.silu(x)


def ref_backward(x, grad_output):
    x.grad = None
    y = F.silu(x)
    y.backward(grad_output)
    return x.grad

