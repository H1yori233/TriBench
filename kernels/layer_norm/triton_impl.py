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


@triton.jit
def _layer_norm_backward_kernel_dX(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    dX_ptr,
    dX_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dy_base = dY_ptr + row_idx * dY_row_stride
    x_base = X_ptr + row_idx * X_row_stride
    dx_base = dX_ptr + row_idx * dX_row_stride

    dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
    X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Re-compute statistics
    mean = tl.sum(X_row, axis=0) / n_cols
    x_centered = tl.where(mask, X_row - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)
    x_hat = x_centered * rstd

    # dX computation
    # dX = (1/sigma) * (gamma * dY - mean(gamma * dY) - x_hat * mean(gamma * dY * x_hat))
    wdy = W_row * dY_row
    mean_wdy = tl.sum(wdy, axis=0) / n_cols
    mean_wdy_xhat = tl.sum(wdy * x_hat, axis=0) / n_cols
    
    dX_row = rstd * (wdy - mean_wdy - x_hat * mean_wdy_xhat)
    tl.store(dx_base + col_offsets, dX_row.to(X_row.dtype), mask=mask)


@triton.jit
def _layer_norm_backward_kernel_dW_dB(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    dW_ptr,
    dB_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dy_base = dY_ptr + row_idx * dY_row_stride
    x_base = X_ptr + row_idx * X_row_stride

    dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
    X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Re-compute statistics
    mean = tl.sum(X_row, axis=0) / n_cols
    x_centered = tl.where(mask, X_row - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)
    x_hat = x_centered * rstd

    # dW = sum(dY * x_hat), dB = sum(dY)
    tl.atomic_add(dW_ptr + col_offsets, dY_row * x_hat, mask=mask)
    tl.atomic_add(dB_ptr + col_offsets, dY_row, mask=mask)



def run(*, X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float, grad_output: torch.Tensor) -> torch.Tensor:
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


def run_backward(*, X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton LayerNorm backward wrapper."""
    shape = X.shape
    dim = shape[-1]
    X_flat = X.view(-1, dim)
    grad_output_flat = grad_output.view(-1, dim)
    n_rows, n_cols = X_flat.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    dX = torch.empty_like(X_flat)
    dW = torch.zeros_like(W, dtype=torch.float32)
    dB = torch.zeros_like(B, dtype=torch.float32)

    _layer_norm_backward_kernel_dX[(n_rows,)](
        grad_output_flat,
        grad_output_flat.stride(0),
        X_flat,
        X_flat.stride(0),
        W,
        dX,
        dX.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    _layer_norm_backward_kernel_dW_dB[(n_rows,)](
        grad_output_flat,
        grad_output_flat.stride(0),
        X_flat,
        X_flat.stride(0),
        dW,
        dB,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return dX.reshape(*shape), dW.to(W.dtype), dB.to(B.dtype)

