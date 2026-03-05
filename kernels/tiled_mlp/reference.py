"""
Reference: full linear in one pass. y = x @ W.t() + b.
Used to compare against tiled (chunked) implementation.
"""
import torch


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 2)
    T = params.get("T", 512)
    H = params.get("H", 1024)
    V = params.get("V", 4096)
    shards = params.get("shards", 4)
    BT = B * T
    x = torch.randn((BT, H), dtype=dtype, device=device, requires_grad=True)
    W = torch.randn((V, H), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((V,), dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn((BT, V), dtype=dtype, device=device)
    return {"x": x, "W": W, "b": b, "shards": shards, "grad_output": grad_output}


def ref(*, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, shards: int, grad_output: torch.Tensor) -> torch.Tensor:
    """Full linear: one pass."""
    return x @ W.t() + b


def ref_backward(*, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, shards: int, grad_output: torch.Tensor):
    x.grad = None
    W.grad = None
    b.grad = None
    y = x @ W.t() + b
    y.backward(grad_output)
    return x.grad, W.grad, b.grad
