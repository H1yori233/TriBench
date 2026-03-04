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
def _fused_add_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,  # output residual
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,  # input residual
    R_row_stride,
    W_ptr,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    S_ptr += row_idx * S_row_stride
    X_ptr += row_idx * X_row_stride
    R_ptr += row_idx * R_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    R_row = tl.load(R_ptr + col_offsets, mask=mask, other=0)
    S_row = X_row + R_row
    tl.store(S_ptr + col_offsets, S_row, mask=mask)
    
    S_row_dtype = S_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    # Simplified casting logic for benchmark
    S_f32 = S_row.to(tl.float32)
    mean_square = tl.sum(S_f32 * S_f32, axis=0) / n_cols
    rstd = tl.math.rsqrt(mean_square + eps)

    if casting_mode == 0: # llama
        S_norm = (S_f32 * rstd).to(S_row_dtype)
        Y_row = S_norm * (offset + W_row).to(S_row_dtype)
    else:
        Y_row = (S_f32 * rstd * (offset + W_row)).to(S_row_dtype)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)

@triton.jit
def _fused_add_rms_norm_backward_kernel_dX(
    DS_ptr,  # output gradient wrt input residual S (unused as output, but we can store it)
    DY_ptr,  # grad wrt Y
    DY_row_stride,
    DX_ptr,  # grad wrt X (also R)
    DX_row_stride,
    DR_ptr,  # grad wrt R
    DR_row_stride,
    GS_ptr,  # grad wrt S (from direct output)
    GS_row_stride,
    S_ptr,   # forward S
    S_row_stride,
    W_ptr,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    DY_ptr += row_idx * DY_row_stride
    DX_ptr += row_idx * DX_row_stride
    DR_ptr += row_idx * DR_row_stride
    GS_ptr += row_idx * GS_row_stride
    S_ptr += row_idx * S_row_stride

    dy = tl.load(DY_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(S_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    gs = tl.load(GS_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    w = w + offset
    
    mean_square = tl.sum(s * s, axis=0) / n_cols
    rstd = tl.math.rsqrt(mean_square + eps)
    
    # dy_norm = dy * w
    dy_w = dy * w
    
    # ds = rstd * (dy_w - s * rstd^2 * mean(s * dy_w))
    # This is the standard RMSNorm backward
    c1 = tl.sum(dy_w * s, axis=0) / n_cols
    ds = rstd * (dy_w - s * (rstd * rstd) * c1)
    
    # Add gradient from direct S output
    ds_total = ds + gs
    
    tl.store(DX_ptr + col_offsets, ds_total.to(DX_ptr.dtype.element_ty), mask=mask)
    tl.store(DR_ptr + col_offsets, ds_total.to(DR_ptr.dtype.element_ty), mask=mask)

@triton.jit
def _fused_add_rms_norm_backward_kernel_dW(
    DY_ptr,
    DY_row_stride,
    S_ptr,
    S_row_stride,
    DW_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    col_idx = tl.program_id(0).to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = col_idx < n_cols
    
    dw = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for i in range(0, n_rows, BLOCK_SIZE_M):
        row_offsets = i + tl.arange(0, BLOCK_SIZE_M)
        mask_m = row_offsets < n_rows
        mask = mask_m[:, None] & mask_n[None, :]
        
        dy = tl.load(DY_ptr + row_offsets[:, None] * DY_row_stride + col_idx[None, :], mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S_ptr + row_offsets[:, None] * S_row_stride + col_idx[None, :], mask=mask, other=0.0).to(tl.float32)
        
        mean_square = tl.sum(s * s, axis=1) / n_cols
        rstd = tl.math.rsqrt(mean_square + eps)
        
        dw += tl.sum(dy * s * rstd[:, None], axis=0)
        
    tl.store(DW_ptr + col_idx, dw.to(DW_ptr.dtype.element_ty), mask_n)


def run(*, X: torch.Tensor, R: torch.Tensor, W: torch.Tensor, eps: float, offset: float = 0.0, casting_mode: str = "llama", grad_output=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton FusedAddRMSNorm wrapper."""
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    R = R.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    S = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    mode_map = {"llama": 0, "gemma": 1, "none": -1}
    c_mode = mode_map.get(casting_mode, 0)

    _fused_add_rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        S,
        S.stride(0),
        X,
        X.stride(0),
        R,
        R.stride(0),
        W,
        n_cols,
        eps,
        offset,
        casting_mode=c_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return Y.view(*shape), S.view(*shape)

def run_backward(*, X: torch.Tensor, R: torch.Tensor, W: torch.Tensor, eps: float, offset: float = 0.0, casting_mode: str = "llama", grad_output=None):
    shape = X.shape
    dim = shape[-1]
    X_f = X.view(-1, dim)
    R_f = R.view(-1, dim)
    n_rows, n_cols = X_f.shape
    
    grad_y, grad_s = grad_output
    grad_y = grad_y.view(-1, n_cols)
    grad_s = grad_s.view(-1, n_cols)
    
    # Need forward S
    _, S = run(X=X, R=R, W=W, eps=eps, offset=offset, casting_mode=casting_mode)
    S_f = S.view(-1, n_cols)
    
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    
    dX = torch.empty_like(X_f)
    dR = torch.empty_like(R_f)
    
    mode_map = {"llama": 0, "gemma": 1, "none": -1}
    c_mode = mode_map.get(casting_mode, 0)

    _fused_add_rms_norm_backward_kernel_dX[(n_rows,)](
        None, # DS_ptr
        grad_y,
        grad_y.stride(0),
        dX,
        dX.stride(0),
        dR,
        dR.stride(0),
        grad_s,
        grad_s.stride(0),
        S_f,
        S_f.stride(0),
        W,
        n_cols,
        eps,
        offset,
        casting_mode=c_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    dW = torch.empty_like(W)
    _fused_add_rms_norm_backward_kernel_dW[(triton.cdiv(n_cols, 64),)](
        grad_y,
        grad_y.stride(0),
        S_f,
        S_f.stride(0),
        dW,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        num_warps=4,
    )
    
    return dX.view(*shape), dR.view(*shape), dW
