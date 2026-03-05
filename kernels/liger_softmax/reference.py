import torch


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 4)
    S = params.get("S", 64)
    H = params.get("H", 128)
    x = torch.randn(B, S, H, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(B, S, H, dtype=dtype, device=device)
    return {"x": x, "grad_output": grad_output}


def ref(*, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def ref_backward(*, x: torch.Tensor, grad_output: torch.Tensor):
    x.grad = None
    y = torch.softmax(x, dim=-1)
    y.backward(grad_output)
    return x.grad
