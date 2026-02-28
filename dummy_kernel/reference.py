import torch

def make_inputs(case: dict, device: str, seed: int, dtype: torch.dtype) -> dict:
    # TODO: Generate dummy inputs
    torch.manual_seed(seed)
    N = case["N"]
    x = torch.randn(N, dtype=dtype, device=device)
    return {"x": x}

def reference(x: torch.Tensor) -> torch.Tensor:
    # TODO: Implement reference logic
    return x.clone()

def estimate(**kwargs) -> dict:
    # TODO: Estimate theoretical FLOPS and memory traffic
    return {"flops": 0.0, "bytes": 0.0}
