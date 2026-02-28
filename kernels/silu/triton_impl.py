import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(
    X, Y,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(X + offsets, mask=mask).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid
    
    tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def silu_backward_kernel(
    dY, X, dX,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    dy = tl.load(dY + offsets, mask=mask).to(tl.float32)
    x = tl.load(X + offsets, mask=mask).to(tl.float32)
    
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    # dX = dY * sigmoid * (1 + X * (1 - sigmoid))
    dx = dy * sigmoid * (1.0 + x * (1.0 - sigmoid))
    
    tl.store(dX + offsets, dx.to(dX.dtype.element_ty), mask=mask)


def run(x, grad_output):
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](
        x, out,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return out


def run_backward(x, grad_output):
    n_elements = x.numel()
    dx = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    silu_backward_kernel[grid](
        grad_output, x, dx,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return dx

