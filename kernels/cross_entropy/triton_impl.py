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


@triton.jit
def liger_cross_entropy_backward_kernel(
    X_ptr,  # Input/Output: stores gradients here
    X_stride,
    Y_ptr,
    Y_stride,
    grad_output_ptr,
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
        # Gradient is 0 for ignored tokens
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    grad_output = tl.load(grad_output_ptr)

    # Re-compute online softmax to get gradients
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # Gradient computation
    # dx_i = (softmax(x_i) - 1(i==y)) / N * grad_output
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).cast(tl.float32)
        
        softmax_X = tl.exp(X_block - m) / d
        
        # dx = softmax(x)
        grad_x = softmax_X
        # if i == y: dx = softmax(x) - 1
        grad_x = tl.where(X_offsets == y, grad_x - 1.0, grad_x)
        
        if reduction == "mean":
            grad_x = grad_x / n_non_ignore
            
        grad_x = grad_x * grad_output
        
        tl.store(X_ptr + X_offsets, grad_x, mask=X_offsets < n_cols)


def run(*, _input: torch.Tensor, target: torch.Tensor, grad_output: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
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


def run_backward(*, _input: torch.Tensor, target: torch.Tensor, grad_output: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Triton CrossEntropy wrapper (Backward only)."""
    BT, V = _input.shape
    n_rows = BT
    
    MAX_FUSED_SIZE = 65536 // 2
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()

    # We modify _input in-place for gradient as is common in Liger-Kernel
    # but for TriBench correctness check, we should probably return a clone or new tensor
    # so we don't mess up the reference check if it's reused.
    grad_input = _input.clone()

    liger_cross_entropy_backward_kernel[(n_rows,)](
        X_ptr=grad_input,
        X_stride=grad_input.stride(0),
        Y_ptr=target,
        Y_stride=target.stride(0),
        grad_output_ptr=grad_output,
        n_cols=V,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        label_smoothing=0.0,
        reduction="mean",
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    return grad_input

