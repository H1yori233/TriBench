import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 4096)
    V = params.get("V", 32000)
    ignore_index = params.get("ignore_index", -100)

    input = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn((V, H), dtype=dtype, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), device=device)
    
    # Randomly set some targets to ignore_index
    mask = torch.rand((B * T,), device=device) < 0.1
    target[mask] = ignore_index

    return {"X": input, "W": weight, "target": target, "ignore_index": ignore_index}


def ref(X: torch.Tensor, W: torch.Tensor, target: torch.Tensor, ignore_index: int):
    logits = X @ W.t()
    return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="mean")
