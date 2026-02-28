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
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_base = Y_ptr + row_idx * Y_row_stride
    x_base = X_ptr + row_idx * X_row_stride

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = tl.math.rsqrt(mean_square + eps)

    Y_row = (X_row * rstd) * W_row
    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel_dX(
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

    dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(x_base + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    # Re-compute rstd
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = tl.math.rsqrt(mean_square + eps)

    # dX = rstd * W * dY - rstd^3 * X * mean(dY * W * X)
    mDYWX = tl.sum(dY_row * W_row * X_row, axis=0) / n_cols
    dX_row = rstd * W_row * dY_row - (rstd * rstd * rstd) * X_row * mDYWX
    
    tl.store(dx_base + col_offsets, dX_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel_dW(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    dW_ptr,
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

    dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(x_base + col_offsets, mask=mask, other=0).to(tl.float32)

    # Re-compute rstd
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = tl.math.rsqrt(mean_square + eps)

    # dW = dY * X * rstd
    dW_row = dY_row * X_row * rstd
    tl.atomic_add(dW_ptr + col_offsets, dW_row, mask=mask)


def run(*, X: torch.Tensor, W: torch.Tensor, eps: float, grad_output: torch.Tensor) -> torch.Tensor:
    """Triton RMSNorm wrapper."""
    shape = X.shape
    dim = shape[-1]
    X_flat = X.view(-1, dim)
    n_rows, n_cols = X_flat.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    _rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X_flat,
        X_flat.stride(0),
        W,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*shape)


def run_backward(*, X: torch.Tensor, W: torch.Tensor, eps: float, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton RMSNorm backward wrapper."""
    shape = X.shape
    dim = shape[-1]
    X_flat = X.view(-1, dim)
    grad_output_flat = grad_output.view(-1, dim)
    n_rows, n_cols = X_flat.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    dX = torch.empty_like(X_flat)
    dW = torch.zeros_like(W, dtype=torch.float32)

    _rms_norm_backward_kernel_dX[(n_rows,)](
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

    _rms_norm_backward_kernel_dW[(n_rows,)](
        grad_output_flat,
        grad_output_flat.stride(0),
        X_flat,
        X_flat.stride(0),
        dW,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return dX.view(*shape), dW.to(W.dtype)

