"""
Reference: causal mask (-inf) -> softmax(dim=-1) -> conv2d (1x1) -> causal mask (zero).
Scores (B, L, L), weight (1, 1, 1, 1), bias (1,) -> output (B, L, L).
"""
import torch
import torch.nn.functional as F


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 2)
    L = params.get("L", 8)
    scores = torch.randn(B, L, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(1, 1, 1, 1, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(1, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(B, L, L, dtype=dtype, device=device)
    return {
        "scores": scores,
        "weight": weight,
        "bias": bias,
        "grad_output": grad_output,
    }


def _causal_mask_inf(x):
    """Set future positions to -1e9 (for softmax). Use float for mask value to avoid fp16 overflow."""
    B, L, _ = x.shape
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
    val = -1e9 if x.dtype != torch.float16 else -1e4
    out = x.masked_fill(mask.unsqueeze(0), val)
    return out


def _causal_mask_zero(x):
    """Set future positions to 0."""
    B, L, _ = x.shape
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
    out = x.masked_fill(mask.unsqueeze(0), 0.0)
    return out


def ref(
    *,
    scores: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    scores_inf = _causal_mask_inf(scores)
    probs = F.softmax(scores_inf.float(), dim=-1).to(scores.dtype)
    probs_4d = probs.unsqueeze(1)
    out_conv = F.conv2d(probs_4d, weight, bias, stride=1, padding=0)
    out = _causal_mask_zero(out_conv.squeeze(1))
    return out


def ref_backward(
    *,
    scores: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
):
    scores.grad = None
    weight.grad = None
    bias.grad = None
    out = ref(scores=scores, weight=weight, bias=bias, grad_output=grad_output)
    out.backward(grad_output)
    return scores.grad, weight.grad, bias.grad
