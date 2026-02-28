import torch
import triton
import triton.language as tl

# Reuse vector_add kernel logic if possible, or just define a fused one
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

def run_fused(x, y):
    n_elements = x.numel()
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_add_silu_kernel[grid](
        x, y, z,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return z

def run_sequential(x, y):
    # Simulating non-fused by calling two separate operations (even if using torch to mock the overhead)
    # In a real scenario, this would be two separate Triton kernels
    # For benchmarking overhead, we can just do it in Torch or call existing kernels
    return torch.nn.functional.silu(x + y)
