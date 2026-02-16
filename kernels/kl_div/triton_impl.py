import torch
import triton
import triton.language as tl

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,
    y_stride,
    gt_ptr,
    gt_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr,
    reduction: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == 0: # NONE
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    if reduction != 0:
        tl.store(loss_ptr, loss_sum)


def run(*, y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str = "batchmean", log_target: bool = False, eps: float = 1e-10) -> torch.Tensor:
    """Triton KLDiv wrapper."""
    BT, V = y_pred.shape
    MAX_FUSED_SIZE = 65536 // 4
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    _str_to_reduction_mode = {"none": 0, "sum": 1, "mean": 2, "batchmean": 3}
    red_mode = _str_to_reduction_mode[reduction]

    out_size = (BT, V) if red_mode == 0 else (BT,)
    output_tensor = torch.zeros(out_size, device=y_pred.device, dtype=torch.float32)

    _kldiv_kernel_forward[(BT,)](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        output_tensor,
        output_tensor.stride(0),
        V,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        log_target=log_target,
        reduction=red_mode,
    )

    if red_mode == 3: # batchmean
        return (output_tensor.sum() / BT).to(y_pred.dtype)
    elif red_mode == 1: # sum
        return output_tensor.sum().to(y_pred.dtype)
    elif red_mode == 2: # mean
        return (output_tensor.sum() / (BT * V)).to(y_pred.dtype)
    else:
        return output_tensor.to(y_pred.dtype)
