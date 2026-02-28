import torch
import torch.nn.functional as F

def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    V = params.get("V", 32000)
    eps = params.get("eps", 1e-10)

    # y_pred is usually log-probabilities
    y_pred = torch.log_softmax(torch.randn((B * T, V), dtype=dtype, device=device), dim=-1).requires_grad_(True)
    # y_true is usually probabilities
    y_true = torch.softmax(torch.randn((B * T, V), dtype=dtype, device=device), dim=-1).requires_grad_(True)
    grad_output = torch.randn((), dtype=torch.float32, device=device)
    
    return {"y_pred": y_pred, "y_true": y_true, "reduction": "batchmean", "log_target": False, "eps": eps, "grad_output": grad_output}


def ref(y_pred, y_true, reduction, log_target, eps, grad_output):
    # PyTorch KLDivLoss expects (input, target)
    return F.kl_div(y_pred, y_true, reduction=reduction, log_target=log_target)


def ref_backward(y_pred, y_true, reduction, log_target, eps, grad_output):
    y_pred.grad = None
    y_true.grad = None
    loss = F.kl_div(y_pred, y_true, reduction=reduction, log_target=log_target)
    loss.backward(grad_output)
    return y_pred.grad, y_true.grad

