import torch

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    H = params.get("H", 4096)
    eps = params.get("eps", 1e-6)

    x = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    residual = torch.randn((B * T, H), dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn((H,), dtype=dtype, device=device, requires_grad=True)
    
    grad_y = torch.randn((B * T, H), dtype=dtype, device=device)
    grad_s = torch.randn((B * T, H), dtype=dtype, device=device)
    
    return {"X": x, "R": residual, "W": weight, "eps": eps, "offset": 0.0, "casting_mode": "llama", "grad_output": (grad_y, grad_s)}


def ref(X, R, W, eps, offset, casting_mode, grad_output):
    # hidden_states = residual + hidden_states
    # residual = hidden_states (returned as S)
    # hidden_states = rmsnorm(hidden_states)
    
    S = X + R
    
    # RMSNorm
    original_dtype = S.dtype
    S_f32 = S.to(torch.float32)
    mean_square = S_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(mean_square + eps)
    
    if casting_mode == "llama":
        # W transformation is often S * (offset + W)
        output = (S_f32 * rstd).to(original_dtype) * (offset + W)
    else:
        output = (S_f32 * rstd * (offset + W)).to(original_dtype)
        
    return output, S

def ref_backward(X, R, W, eps, offset, casting_mode, grad_output):
    X.grad = None
    R.grad = None
    W.grad = None
    Y, S = ref(X, R, W, eps, offset, casting_mode, grad_output)
    torch.autograd.backward([Y, S], [grad_output[0], grad_output[1]])
    return X.grad, R.grad, W.grad

