import torch


def make_inputs(case: dict, device: str, seed: int, dtype: torch.dtype) -> dict:
    """Generate input tensors for vector addition."""
    torch.manual_seed(seed)
    N = case["N"]
    x = torch.randn(N, device=device, dtype=dtype)
    y = torch.randn(N, device=device, dtype=dtype)
    return {"x": x, "y": y}


def ref(*, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: element-wise addition."""
    return x + y


def estimate(case: dict) -> dict:
    """Estimate FLOPs and memory bytes for vector addition."""
    N = case["N"]
    # 1 add per element
    flops = N
    # Read x, y; write z: each element is typically 2 bytes (fp16) or 4 bytes (fp32)
    # Use fp16 (2 bytes) as conservative estimate; actual depends on dtype
    bytes_ = 3 * N * 2
    return {"flops": flops, "bytes": bytes_}
