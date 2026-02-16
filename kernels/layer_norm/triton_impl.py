import torch
import triton
import triton.language as tl

def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    B_ptr,  # pointer to bias, shape (n_cols,)
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Calculate pointers for this row
    row_X_ptr = X_ptr + row_idx * X_row_stride
    row_Y_ptr = Y_ptr + row_idx * Y_row_stride

    # Load input data and convert to fp32 for numerical stability
    X_row = tl.load(row_X_ptr + col_offsets, mask=mask, other=0.0)
    X_f32 = X_row.to(tl.float32)

    # Compute statistics in fp32 for numerical stability
    mean = tl.sum(X_f32, axis=0) / n_cols
    X_centered = X_f32 - mean
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)

    # Pre-load weights and bias
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Y = (X - mean) * rstd * W + B
    Y_f32 = X_centered * rstd * W_row + B_row

    # Store output
    tl.store(row_Y_ptr + col_offsets, Y_f32.to(X_row.dtype), mask=mask)


def run(*, X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float) -> torch.Tensor:
    """Triton LayerNorm wrapper."""
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    _layer_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        B,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*shape)
