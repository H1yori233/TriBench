import torch

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    M = params.get("M", 2048)
    N = params.get("N", 2048)
    K = params.get("K", 2048)

    A = torch.rand((M, K), dtype=dtype, device=device).requires_grad_(True)
    B = torch.rand((K, N), dtype=dtype, device=device).requires_grad_(True)
    grad_output = torch.randn((M, N), dtype=dtype, device=device)
    
    return {"a": A, "b": B, "grad_output": grad_output}


def ref(a, b, grad_output):
    return torch.mm(a, b)

def ref_backward(a, b, grad_output):
    a.grad = None
    b.grad = None
    c = torch.mm(a, b)
    c.backward(grad_output)
    return a.grad, b.grad

