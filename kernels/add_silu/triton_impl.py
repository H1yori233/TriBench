import torch
import triton
import triton.language as tl

@triton.jit
def _fused_add_silu_kernel(
    x_ptr, y_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Fused operation
    res = x + y
    # SiLU: x * sigmoid(x)
    res = res * tl.sigmoid(res)
    
    tl.store(z_ptr + offsets, res, mask=mask)

@triton.jit
def _fused_add_silu_backward_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, dy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    dz = tl.load(dz_ptr + offsets, mask=mask).to(tl.float32)
    
    a = x + y
    sigmoid_a = tl.sigmoid(a)
    # d_silu(a)/da = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    da = dz * sigmoid_a * (1.0 + a * (1.0 - sigmoid_a))
    
    tl.store(dx_ptr + offsets, da.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dy_ptr + offsets, da.to(dy_ptr.dtype.element_ty), mask=mask)

def run_fused(x, y, grad_output):
    n_elements = x.numel()
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_add_silu_kernel[grid](
        x, y, z,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return z

def run_backward(x, y, grad_output):
    n_elements = x.numel()
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_add_silu_backward_kernel[grid](
        x, y, grad_output, dx, dy,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return dx, dy

def run_sequential(x, y, grad_output=None):
    return torch.nn.functional.silu(x + y)
