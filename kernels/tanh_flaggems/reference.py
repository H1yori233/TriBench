"""
Reference: element-wise tanh. Matches FlagGems tanh semantics.
"""
import torch


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype) -> dict:
    torch.manual_seed(seed)
    N = params["N"]
    x = torch.randn(N, device=device, dtype=dtype).requires_grad_(True)
    grad_output = torch.randn(N, device=device, dtype=dtype)
    return {"x": x, "grad_output": grad_output}


def ref(*, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def ref_backward(*, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    x.grad = None
    y = torch.tanh(x)
    y.backward(grad_output)
    return x.grad


def estimate(params: dict) -> dict:
    N = params["N"]
    flops = 2 * N  # tanh approx
    bytes_ = 2 * N * 2  # read x, write y
    return {"flops": flops, "bytes": bytes_}
