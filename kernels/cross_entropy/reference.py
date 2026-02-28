import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    V = params.get("V", 32000)
    ignore_index = params.get("ignore_index", -100)

    input = torch.randn((B * T, V), dtype=dtype, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), device=device)
    
    # Randomly set some targets to ignore_index
    mask = torch.rand((B * T,), device=device) < 0.1
    target[mask] = ignore_index

    # For backward
    grad_output = torch.randn((), dtype=torch.float32, device=device)

    return {
        "_input": input, 
        "target": target, 
        "ignore_index": ignore_index,
        "grad_output": grad_output
    }


def ref(_input: torch.Tensor, target: torch.Tensor, ignore_index: int, grad_output: torch.Tensor):
    # PyTorch CrossEntropy expects (N, C) or (N, d1, d2, ..., K, C)
    # Our input is (BT, V), target is (BT)
    return F.cross_entropy(_input, target, ignore_index=ignore_index, reduction="mean")


def ref_backward(_input: torch.Tensor, target: torch.Tensor, ignore_index: int, grad_output: torch.Tensor):
    _input.grad = None
    loss = F.cross_entropy(_input, target, ignore_index=ignore_index, reduction="mean")
    loss.backward(grad_output)
    return _input.grad

