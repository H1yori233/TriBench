import torch

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 64)
    M = params.get("M", 128)
    N = params.get("N", 128)
    K = params.get("K", 128)
    alpha = params.get("alpha", 1.0)
    beta = params.get("beta", 1.0)

    A = torch.randn((B, M, K), dtype=dtype, device=device, requires_grad=True)
    B_mat = torch.randn((B, K, N), dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn((B, M, N), dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn((B, M, N), dtype=dtype, device=device)
    
    return {"bias": bias, "A": A, "B": B_mat, "beta": beta, "alpha": alpha, "grad_output": grad_output}


def ref(bias, A, B, beta, alpha, grad_output):
    return torch.baddbmm(bias, A, B, beta=beta, alpha=alpha)

def ref_backward(bias, A, B, beta, alpha, grad_output):
    A.grad = None
    B.grad = None
    bias.grad = None
    C = torch.baddbmm(bias, A, B, beta=beta, alpha=alpha)
    C.backward(grad_output)
    return bias.grad, A.grad, B.grad

