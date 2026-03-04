import torch


def make_inputs(case: dict, device: str, seed: int, dtype: torch.dtype) -> dict:
    """Generate a 3-D input tensor (B, S, H) for softmax."""
    torch.manual_seed(seed)
    B, S, H = case["B"], case["S"], case["H"]
    x = torch.randn(B, S, H, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(B, S, H, device=device, dtype=dtype)
    return {"x": x, "grad_output": grad_output}


def ref(*, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: softmax over the last dimension."""
    return torch.softmax(x, dim=-1)


def ref_backward(*, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: gradient of softmax."""
    x.grad = None
    y = torch.softmax(x, dim=-1)
    y.backward(grad_output)
    return x.grad



def estimate(case: dict) -> dict:
    """Estimate FLOPs and bytes for softmax.

    Softmax ≈ 5N FLOPs (exp, sum, div per element + max subtraction).
    Memory: read x, write y => 2 * total_elements * element_size.
    """
    B, S, H = case["B"], case["S"], case["H"]
    total = B * S * H
    flops = 5 * total
    # Assume fp16 (2 bytes) as a baseline
    bytes_ = 2 * total * 2
    return {"flops": flops, "bytes": bytes_}
