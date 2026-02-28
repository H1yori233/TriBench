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
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    da_ptr, db_ptr, dy_ptr, a_ptr, b_ptr, 
    stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    da_ptr += program_id * stride
    db_ptr += program_id * stride
    dy_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dy_row = tl.load(dy_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU(a) = a * sigmoid(a)
    sigmoid_a = tl.sigmoid(a_row)
    silu_a = a_row * sigmoid_a
    
    # da = dy * silu'(a) * b = dy * sigmoid(a) * (1 + a * (1 - sigmoid(a))) * b
    da_row = dy_row * (sigmoid_a * (1.0 + a_row * (1.0 - sigmoid_a))) * b_row
    # db = dy * silu(a)
    db_row = dy_row * silu_a
    
    tl.store(da_ptr + col_offsets, da_row.to(da_ptr.dtype.element_ty), mask=mask)
    tl.store(db_ptr + col_offsets, db_row.to(db_ptr.dtype.element_ty), mask=mask)



def run(*, a: torch.Tensor, b: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Triton SwiGLU wrapper."""
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.view(*ori_shape)


def run_backward(*, a: torch.Tensor, b: torch.Tensor, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton SwiGLU backward wrapper."""
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    grad_output = grad_output.view(-1, n_cols)
    
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        da,
        db,
        grad_output,
        a,
        b,
        da.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return da.view(*ori_shape), db.view(*ori_shape)

