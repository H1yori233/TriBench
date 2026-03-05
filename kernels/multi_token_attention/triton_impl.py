"""
Multi-token attention: Triton causal mask + PyTorch softmax + conv2d + Triton mask.
Softmax path only (no sparsemax). From temp_for_reference/multi_token_attention.py.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        BLOCK_SIZE = MAX_FUSED_SIZE
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _mask_fwd_kernel(
    scores_ptr,
    out_ptr,
    stride_b,
    stride_m,
    stride_n,
    L,
    mask_val: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)
    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)
    base = scores_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    future = col_idx[None, :] > row_idx[:, None]
    mask_load = in_bounds & ~future
    out = tl.load(base + offs, mask=mask_load, other=mask_val)
    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds)


@triton.jit
def _mask_bwd_kernel(
    grad_in_ptr,
    out_ptr,
    stride_b,
    stride_m,
    stride_n,
    L,
    BLOCK: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)
    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)
    base = grad_in_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    grad_vals = tl.load(base + offs, mask=in_bounds, other=0.0)
    future = col_idx[None, :] > row_idx[:, None]
    zero = tl.zeros(grad_vals.shape, dtype=grad_vals.dtype)
    out = tl.where(future, zero, grad_vals)
    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds)


def _mask_inf_forward(scores: torch.Tensor, mask_val: float = -1e9) -> torch.Tensor:
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.contiguous().view(N, L, L)
    out = torch.empty_like(scores_f)
    sb, sm, sn = scores_f.stride(0), scores_f.stride(1), scores_f.stride(2)
    BLOCK_SIZE, num_warps = calculate_settings(L)
    grid = (triton.cdiv(L, BLOCK_SIZE), triton.cdiv(L, BLOCK_SIZE), N)
    _mask_fwd_kernel[grid](scores_f, out, sb, sm, sn, L, mask_val=mask_val, BLOCK=BLOCK_SIZE)
    return out.view(*batch, L, L)


def _mask_zero_forward(scores: torch.Tensor) -> torch.Tensor:
    return _mask_inf_forward(scores, mask_val=0.0)


def _mask_inf_backward(grad: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = grad.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    grad_f = grad.contiguous().view(N, L, L)
    out = torch.empty_like(grad_f)
    sb, sm, sn = grad_f.stride(0), grad_f.stride(1), grad_f.stride(2)
    BLOCK_SIZE, num_warps = calculate_settings(L)
    grid = (triton.cdiv(L, BLOCK_SIZE), triton.cdiv(L, BLOCK_SIZE), N)
    _mask_bwd_kernel[grid](grad_f, out, sb, sm, sn, L, BLOCK=BLOCK_SIZE)
    return out.view(*batch, L, L)


def _mask_zero_backward(grad: torch.Tensor) -> torch.Tensor:
    return _mask_inf_backward(grad)


def run(
    *,
    scores: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    mask_val = -1e4 if scores.dtype == torch.float16 else -1e9
    scores_inf = _mask_inf_forward(scores, mask_val=mask_val)
    probs = F.softmax(scores_inf.float(), dim=-1).to(scores.dtype)
    probs_4d = probs.unsqueeze(1)
    out_conv = F.conv2d(probs_4d, weight, bias, stride=1, padding=0)
    out = _mask_zero_forward(out_conv.squeeze(1))
    return out


def run_backward(
    *,
    scores: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
):
    mask_val = -1e4 if scores.dtype == torch.float16 else -1e9
    scores_inf = _mask_inf_forward(scores, mask_val=mask_val)
    probs = F.softmax(scores_inf.float(), dim=-1).to(scores.dtype)
    probs_4d = probs.unsqueeze(1)
    out_conv = F.conv2d(probs_4d, weight, bias, stride=1, padding=0)
    out = _mask_zero_forward(out_conv.squeeze(1))

    grad_conv = _mask_zero_backward(grad_output).unsqueeze(1)
    grad_probs = F.conv_transpose2d(grad_conv, weight, None, stride=1, padding=0)
    grad_probs = grad_probs.squeeze(1)
    grad_weight = torch.nn.grad.conv2d_weight(
        probs_4d, weight.shape, grad_conv, stride=1, padding=0
    )
    grad_bias = grad_conv.sum(dim=(0, 2, 3))
    grad_scores_float = probs * (grad_probs.float() - (grad_probs.float() * probs).sum(dim=-1, keepdim=True))
    grad_scores_inf = _mask_inf_backward(grad_scores_float.to(scores.dtype))
    return grad_scores_inf, grad_weight, grad_bias
