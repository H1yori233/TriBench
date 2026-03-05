import torch


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 4096)
    eps = params.get("eps", 1e-6)

    x = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn((H,), dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn((B * T, H), dtype=dtype, device=device)
    return {"X": x, "W": weight, "eps": eps, "grad_output": grad_output}


def ref(X: torch.Tensor, W: torch.Tensor, eps: float, grad_output: torch.Tensor):
    original_dtype = X.dtype
    X = X.to(torch.float32)
    W = W.to(torch.float32)

    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance + eps)
    output = X * W
    return output.to(original_dtype)


def ref_backward(X: torch.Tensor, W: torch.Tensor, eps: float, grad_output: torch.Tensor):
    X.grad = None
    W.grad = None
    y = ref(X, W, eps, grad_output)
    y.backward(grad_output)
    return X.grad, W.grad

