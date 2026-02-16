import torch
import triton
import triton.language as tl

@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        return

    loss_ptr += program_id * loss_stride

    # Online softmax
    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)
    loss = lse - ori_X_y

    if reduction == "mean":
        loss = loss / n_non_ignore

    tl.store(loss_ptr, loss)


def run(*, _input: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Triton CrossEntropy wrapper (Forward only)."""
    BT, V = _input.shape
    n_rows = BT
    
    # Simple MAX_FUSED_SIZE for compatibility
    MAX_FUSED_SIZE = 65536 // 2
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
    
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()

    liger_cross_entropy_kernel[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(0),
        Y_ptr=target,
        Y_stride=target.stride(0),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(0),
        n_cols=V,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        label_smoothing=0.0,
        reduction="mean",
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    return torch.sum(loss_1d).to(_input.dtype)
