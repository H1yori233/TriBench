"""
FlagGems-style tanh: element-wise tanh with float32 promotion for stability.
Replaces pointwise_dynamic / tl_extra_shim with explicit grid and tl.extra.cuda.libdevice.tanh.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.extra.cuda.libdevice.tanh(x)
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tanh_backward_kernel(
    y_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)
    # d/dx tanh(x) = 1 - tanh(x)^2
    dx = dy * (1.0 - y * y)
    tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def run(*, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    y = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, 1024),)
    _tanh_kernel[grid](x, y, n_elements=n_elements, BLOCK_SIZE=1024)
    return y


def run_backward(*, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    y = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, 1024),)
    _tanh_kernel[grid](x, y, n_elements=n_elements, BLOCK_SIZE=1024)
    dx = torch.empty_like(x)
    _tanh_backward_kernel[grid](y, grad_output, dx, n_elements=n_elements, BLOCK_SIZE=1024)
    return dx
