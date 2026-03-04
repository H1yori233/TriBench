import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    V = params.get("V", 32000)
    beta = params.get("beta", 0.5)

    # JSD expects inputs in log-space
    input = torch.log_softmax(torch.randn((B * T, V), dtype=dtype, device=device), dim=-1).requires_grad_(True)
    target = torch.log_softmax(torch.randn((B * T, V), dtype=dtype, device=device), dim=-1).requires_grad_(True)
    grad_output = torch.randn((), dtype=torch.float32, device=device)
    
    return {"_input": input, "target": target, "beta": beta, "grad_output": grad_output}


def ref(_input, target, beta, grad_output):
    # JSD(P || Q) = beta * KL(P || M) + (1-beta) * KL(Q || M)
    # where M = beta * P + (1-beta) * Q
    # In Liger: _input = log Q, target = log P
    
    P = torch.exp(target)
    Q = torch.exp(_input)
    M = beta * P + (1 - beta) * Q
    log_M = torch.log(M)
    
    kl_pm = (P * (target - log_M)).sum() / _input.shape[0]
    kl_qm = (Q * (_input - log_M)).sum() / _input.shape[0]
    
    return beta * kl_pm + (1 - beta) * kl_qm

def ref_backward(_input, target, beta, grad_output):
    _input.grad = None
    target.grad = None
    
    P = torch.exp(target)
    Q = torch.exp(_input)
    M = beta * P + (1 - beta) * Q
    log_M = torch.log(M)
    
    kl_pm = (P * (target - log_M)).sum() / _input.shape[0]
    kl_qm = (Q * (_input - log_M)).sum() / _input.shape[0]
    
    loss = beta * kl_pm + (1 - beta) * kl_qm
    loss.backward(grad_output)
    return _input.grad, target.grad

