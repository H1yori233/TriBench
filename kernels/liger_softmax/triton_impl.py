"""
Liger-Kernel softmax: single-block and multi-block Triton kernels.
No liger_kernel dependency; local calculate_settings.
"""
import torch
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
def _softmax_single_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(X_ptr + row_id * X_row_stride + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)
    y = e / d
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask)


@triton.jit
def _softmax_multi_block_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    m = tl.float32(-float("inf"))
    d = tl.float32(0.0)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"))
        blk_max = tl.max(xblk, axis=0)
        new_m = tl.max(m, blk_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
        m = new_m
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"))
        yblk = tl.exp(xblk - m) / d
        tl.store(Y_ptr + row_id * Y_row_stride + idx, yblk, mask=mask)


@triton.jit
def _softmax_single_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    dy = tl.load(dy_ptr + row_id * dy_stride + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_id * y_stride + offs, mask=mask, other=0.0)
    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)
    tl.store(dx_ptr + row_id * dx_stride + offs, dx, mask=mask)


@triton.jit
def _softmax_multi_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.float32(0.0)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
        y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=0.0)
        acc += tl.sum(dy_blk * y_blk, axis=0)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
        y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=0.0)
        dx_blk = y_blk * (dy_blk - acc)
        tl.store(dx_ptr + row_id * dx_stride + idx, dx_blk, mask=mask)


def run(*, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y2d = torch.empty_like(x2d)
    if n_cols <= BLOCK_SIZE:
        _softmax_single_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
    else:
        _softmax_multi_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
    return y2d.view(*batch, n_cols)


def run_backward(*, x: torch.Tensor, grad_output: torch.Tensor):
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y2d = torch.empty_like(x2d)
    dx2d = torch.empty_like(x2d)
    if n_cols <= BLOCK_SIZE:
        _softmax_single_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
        _softmax_single_block_backward_kernel[(n_rows,)](
            grad_output.view(-1, n_cols), grad_output.stride(0),
            y2d, y2d.stride(0), dx2d, dx2d.stride(0),
            n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
    else:
        _softmax_multi_block_forward_kernel[(n_rows,)](
            y2d, y2d.stride(0), x2d, x2d.stride(0), n_cols,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
        _softmax_multi_block_backward_kernel[(n_rows,)](
            grad_output.view(-1, n_cols), grad_output.stride(0),
            y2d, y2d.stride(0), dx2d, dx2d.stride(0),
            n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
    return dx2d.view(*batch, n_cols)
