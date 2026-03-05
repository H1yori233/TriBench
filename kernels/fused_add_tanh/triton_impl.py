"""
Fused Add + Tanh: one kernel for z = tanh(x + y).
Backward: dz/dx = dz/dy = grad_output * (1 - z^2).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_tanh_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    a = x + y
    z = tl.extra.cuda.libdevice.tanh(a)
    tl.store(z_ptr + offsets, z, mask=mask)


@triton.jit
def _fused_add_tanh_backward_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    dz_ptr,
    dx_ptr,
    dy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    z = tl.load(z_ptr + offsets, mask=mask).to(tl.float32)
    dz = tl.load(dz_ptr + offsets, mask=mask).to(tl.float32)
    # d/dx tanh(x+y) = (1 - z^2) * 1, same for dy
    da = dz * (1.0 - z * z)
    tl.store(dx_ptr + offsets, da.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dy_ptr + offsets, da.to(dy_ptr.dtype.element_ty), mask=mask)


def run_fused(*, x: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    z = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, 1024),)
    _fused_add_tanh_kernel[grid](x, y, z, n_elements=n_elements, BLOCK_SIZE=1024)
    return z


def run_backward(*, x: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor):
    n_elements = x.numel()
    z = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, 1024),)
    _fused_add_tanh_kernel[grid](x, y, z, n_elements=n_elements, BLOCK_SIZE=1024)
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    _fused_add_tanh_backward_kernel[grid](
        x, y, z, grad_output, dx, dy,
        n_elements=n_elements, BLOCK_SIZE=1024,
    )
    return dx, dy


def run_sequential(*, x: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x + y)
