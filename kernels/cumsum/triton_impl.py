import torch
import triton
import triton.language as tl

@triton.jit
def cumsum_kernel(
    X, Y,
    N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    TILE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    
    x_ptr = X + pid_m * stride_xm + offsets * stride_xn
    y_ptr = Y + pid_m * stride_ym + offsets * stride_yn
    
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    y = tl.cumsum(x, axis=0)
    
    tl.store(y_ptr, y.to(Y.dtype.element_ty), mask=mask)


def run(x, dim=-1):
    if dim != -1 and dim != x.ndim - 1:
        # For simplicity, we only support last dim in this benchmark implementation
        # Standard FlagGems implementation handles all dims.
        return torch.cumsum(x, dim=dim)
        
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    
    out = torch.empty_like(x)
    
    TILE_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    
    num_warps = 4
    if TILE_SIZE > 2048:
        num_warps = 8
    if TILE_SIZE > 8192:
        num_warps = 16

    cumsum_kernel[grid](
        x, out,
        N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
    )
    return out
