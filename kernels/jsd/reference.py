import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    V = params.get("V", 32000)
    beta = params.get("beta", 0.5)

    # JSD expects inputs in log-space
    input = torch.log_softmax(torch.randn((B * T, V), dtype=dtype, device=device), dim=-1)
    target = torch.log_softmax(torch.randn((B * T, V), dtype=dtype, device=device), dim=-1)
    
    return {"_input": input, "target": target, "beta": beta}


def ref(_input, target, beta):
    # JSD(P || Q) = beta * KL(P || M) + (1-beta) * KL(Q || M)
    # where M = beta * P + (1-beta) * Q
    # In Liger: _input = log Q, target = log P
    
    P = torch.exp(target)
    Q = torch.exp(_input)
    M = beta * P + (1 - beta) * Q
    log_M = torch.log(M)
    
    kl_pm = F.kl_div(_input, M, reduction='batchmean', log_target=False) # Wait, KL(P||M)
    # KL(P||M) = sum(P * (log P - log M))
    kl_pm = (P * (target - log_M)).sum() / _input.shape[0]
    kl_qm = (Q * (_input - log_M)).sum() / _input.shape[0]
    
    return beta * kl_pm + (1 - beta) * kl_qm
