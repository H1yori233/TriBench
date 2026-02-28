import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 14336)

    a = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn((B * T, H), dtype=dtype, device=device)
    return {"a": a, "b": b, "grad_output": grad_output}


def ref(a, b, grad_output):
    # PyTorch GELU default is 'none' approximation, but Liger uses 'tanh'
    return F.gelu(a, approximate='tanh') * b


def ref_backward(a, b, grad_output):
    a.grad = None
    b.grad = None
    y = F.gelu(a, approximate='tanh') * b
    y.backward(grad_output)
    return a.grad, b.grad

