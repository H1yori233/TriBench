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
    grad_output = torch.randn((B * T, H), dtype=dtype, device=device)
    
    return {"a": a, "b": b, "grad_output": grad_output}


def ref(a: torch.Tensor, b: torch.Tensor, grad_output: torch.Tensor):
    """Reference implementation of SwiGLU."""
    return F.silu(a) * b


def ref_backward(a: torch.Tensor, b: torch.Tensor, grad_output: torch.Tensor):
    """Reference backward implementation of SwiGLU."""
    a.grad = None
    b.grad = None
    y = F.silu(a) * b
    y.backward(grad_output)
    return a.grad, b.grad

