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

@triton.jit
def cumsum_backward_kernel(
    DY, DX,
    N,
    stride_dym, stride_dyn,
    stride_dxm, stride_dxn,
    TILE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    
    dy_ptr = DY + pid_m * stride_dym + offsets * stride_dyn
    dx_ptr = DX + pid_m * stride_dxm + offsets * stride_dxn
    
    dy = tl.load(dy_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Reverse cumsum: dx_i = sum_{j=i}^N dy_j
    total_sum = tl.sum(dy, axis=0)
    cs = tl.cumsum(dy, axis=0)
    dx = total_sum - cs + dy
    
    tl.store(dx_ptr, dx.to(DX.dtype.element_ty), mask=mask)


def run(x, dim=-1, grad_output=None):
    if dim != -1 and dim != x.ndim - 1:
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

def run_backward(x, dim=-1, grad_output=None):
    if dim != -1 and dim != x.ndim - 1:
        # Standard torch backward for unsupported dims
        x.grad = None
        y = torch.cumsum(x, dim=dim)
        y.backward(grad_output)
        return x.grad
        
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    
    dx = torch.empty_like(x)
    TILE_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    
    num_warps = 4
    if TILE_SIZE > 2048:
        num_warps = 8
    if TILE_SIZE > 8192:
        num_warps = 16

    cumsum_backward_kernel[grid](
        grad_output, dx,
        N,
        grad_output.stride(0), grad_output.stride(1),
        dx.stride(0), dx.stride(1),
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
    )
    return dx
