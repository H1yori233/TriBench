"""
Reference: Add + Tanh. z = tanh(x + y).
Same interface as add_silu; used to compare fused vs sequential.
"""
import torch
import torch.nn.functional as F


def make_inputs(params: dict, device: str, seed: int, dtype: str):
    torch.manual_seed(seed)
    N = params["N"]
    torch_dtype = torch.float16 if dtype == "fp16" else (torch.bfloat16 if dtype == "bf16" else torch.float32)
    x = torch.randn(N, device=device, dtype=torch_dtype).requires_grad_(True)
    y = torch.randn(N, device=device, dtype=torch_dtype).requires_grad_(True)
    grad_output = torch.randn(N, device=device, dtype=torch_dtype)
    return {"x": x, "y": y, "grad_output": grad_output}


def ref(*, x: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x + y)


def ref_backward(*, x: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor):
    x.grad = None
    y.grad = None
    z = torch.tanh(x + y)
    z.backward(grad_output)
    return x.grad, y.grad


def estimate(params: dict) -> dict:
    N = params["N"]
    # add: N, tanh: ~2N (approx)
    flops = 3 * N
    bytes_ = 3 * N * 2
    return {"flops": flops, "bytes": bytes_}
