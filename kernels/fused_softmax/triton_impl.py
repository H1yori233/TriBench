import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused softmax kernel: one program per row."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

    # numerically stable softmax
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def run(*, x: torch.Tensor) -> torch.Tensor:
    """Triton fused softmax over the last dimension.

    Input shape: (..., H), flattened to (n_rows, H) internally.
    """
    assert x.is_contiguous(), "Input must be contiguous"
    orig_shape = x.shape
    n_cols = x.shape[-1]
    x_2d = x.reshape(-1, n_cols)
    n_rows = x_2d.shape[0]

    # Block size must be power-of-two >= n_cols (for masking/padding)
    BLOCK_SIZE = _next_power_of_2(n_cols)

    output = torch.empty_like(x_2d)
    softmax_kernel[(n_rows,)](
        x_2d,
        output,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(orig_shape)
