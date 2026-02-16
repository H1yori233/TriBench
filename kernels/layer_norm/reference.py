import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 4096)
    eps = params.get("eps", 1e-6)

    x = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn((H,), dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn((H,), dtype=dtype, device=device, requires_grad=True)
    return {"X": x, "W": weight, "B": bias, "eps": eps}


def ref(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float):
    return F.layer_norm(X, (X.shape[-1],), weight=W, bias=B, eps=eps)
