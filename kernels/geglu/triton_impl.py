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
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # tanh approximation form of GELU
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tl.extra.cuda.libdevice.tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a.to(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel(
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

    # tanh approximation form of GELU
    # GELU(a) = 0.5 * a * (1 + tanh(z))
    # z = sqrt(2/pi) * (a + 0.044715 * a^3)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    z = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_z = tl.extra.cuda.libdevice.tanh(z)
    
    # GELU'(a) = 0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * dz/da
    # dz/da = sqrt_2_over_pi * (1 + 3 * 0.044715 * a^2)
    dz_da = sqrt_2_over_pi * (1.0 + 0.134145 * a_row * a_row)
    gelu_prime_a = 0.5 * (1.0 + tanh_z) + 0.5 * a_row * (1.0 - tanh_z * tanh_z) * dz_da
    
    # da = dy * GELU'(a) * b
    da_row = dy_row * gelu_prime_a * b_row
    # db = dy * GELU(a)
    db_row = dy_row * (0.5 * a_row * (1.0 + tanh_z))
    
    tl.store(da_ptr + col_offsets, da_row.to(da_ptr.dtype.element_ty), mask=mask)
    tl.store(db_ptr + col_offsets, db_row.to(db_ptr.dtype.element_ty), mask=mask)



def run(*, a: torch.Tensor, b: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Triton GEGLU wrapper."""
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_forward_kernel[(n_rows,)](
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
    """Triton GEGLU backward wrapper."""
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a_flat = a.view(-1, n_cols)
    b_flat = b.view(-1, n_cols)
    grad_output_flat = grad_output.view(-1, n_cols)
    
    da = torch.empty_like(a_flat)
    db = torch.empty_like(b_flat)
    n_rows = a_flat.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_backward_kernel[(n_rows,)](
        da,
        db,
        grad_output_flat,
        a_flat,
        b_flat,
        da.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return da.view(*ori_shape), db.view(*ori_shape)

