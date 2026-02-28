import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def run(*, x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, grad_output=None) -> torch.Tensor:
    """Triton vector addition wrapper."""
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output

def run_backward(*, x, y, grad_output, block_size: int = 1024):
    # dx = dz, dy = dz
    return grad_output, grad_output

