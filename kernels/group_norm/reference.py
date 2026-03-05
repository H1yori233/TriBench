import torch
import torch.nn.functional as F


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    C = params.get("C", 4096)
    num_groups = params.get("num_groups", 32)
    eps = params.get("eps", 1e-6)
    assert C % num_groups == 0

    # X: (B, C, T) for group_norm over channels
    X = torch.randn((B, C, T), dtype=dtype, device=device, requires_grad=True)
    W = torch.randn((C,), dtype=dtype, device=device, requires_grad=True)
    B_t = torch.randn((C,), dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn((B, C, T), dtype=dtype, device=device)
    return {"X": X, "W": W, "B": B_t, "num_groups": num_groups, "eps": eps, "grad_output": grad_output}


def ref(
    *,
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor,
    num_groups: int,
    eps: float,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return F.group_norm(X, num_groups, weight=W, bias=B, eps=eps)


def ref_backward(
    *,
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor,
    num_groups: int,
    eps: float,
    grad_output: torch.Tensor,
):
    X.grad = None
    W.grad = None
    B.grad = None
    y = F.group_norm(X, num_groups, weight=W, bias=B, eps=eps)
    y.backward(grad_output)
    return X.grad, W.grad, B.grad
