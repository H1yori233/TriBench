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
        S_row = (S_f32 * rstd).to(S_row_dtype)
        Y_row = S_row * (offset + W_row).to(S_row_dtype)
    else:
        Y_row = (S_f32 * rstd * (offset + W_row)).to(S_row_dtype)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


def run(*, X: torch.Tensor, R: torch.Tensor, W: torch.Tensor, eps: float, offset: float = 0.0, casting_mode: str = "llama") -> tuple[torch.Tensor, torch.Tensor]:
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
