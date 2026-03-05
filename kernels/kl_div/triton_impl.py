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


@triton.jit
def _kldiv_kernel_backward(
    y_ptr,
    y_stride,
    gt_ptr,
    gt_stride,
    dy_pred_ptr,
    dy_pred_stride,
    dy_true_ptr,
    dy_true_stride,
    grad_output_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr,
    reduction: tl.constexpr,
    BT: tl.constexpr,
    V: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    dy_pred_ptr += pid * dy_pred_stride
    dy_true_ptr += pid * dy_true_stride

    # grad_output is typically a scalar for KLDiv (since output is a scalar loss)
    grad_output = tl.load(grad_output_ptr)
    
    # Normalization factor
    norm = 1.0
    if reduction == 3: # batchmean
        norm = 1.0 / BT
    elif reduction == 2: # mean
        norm = 1.0 / (BT * V)

    base_offsets = tl.arange(0, BLOCK_SIZE)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # dLoss/dy_pred = -target (if log_target=False) or -exp(target) (if log_target=True)
        if not log_target:
            dy_pred = -y_true
            # dLoss/dy_true = log(y_true) - y + 1
            dy_true = tl.log(tl.maximum(y_true, eps)) - y + 1.0
        else:
            exp_gt = tl.exp(y_true)
            dy_pred = -exp_gt
            # dLoss/dy_true = exp(target) * (target - input + 1)
            dy_true = exp_gt * (y_true - y + 1.0)

        tl.store(dy_pred_ptr + offsets, (dy_pred * grad_output * norm).to(dy_pred_ptr.dtype.element_ty), mask=mask)
        tl.store(dy_true_ptr + offsets, (dy_true * grad_output * norm).to(dy_true_ptr.dtype.element_ty), mask=mask)



def run(*, y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str = "batchmean", log_target: bool = False, eps: float = 1e-10, grad_output: torch.Tensor) -> torch.Tensor:
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


def run_backward(*, y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str = "batchmean", log_target: bool = False, eps: float = 1e-10, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton KLDiv backward wrapper."""
    BT, V = y_pred.shape
    MAX_FUSED_SIZE = 65536 // 4
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    _str_to_reduction_mode = {"none": 0, "sum": 1, "mean": 2, "batchmean": 3}
    red_mode = _str_to_reduction_mode[reduction]

    dy_pred = torch.empty_like(y_pred)
    dy_true = torch.empty_like(y_true)

    _kldiv_kernel_backward[(BT,)](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        dy_pred,
        dy_pred.stride(0),
        dy_true,
        dy_true.stride(0),
        grad_output,
        V,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        log_target=log_target,
        reduction=red_mode,
        BT=BT,
        V=V,
    )
    return dy_pred, dy_true

