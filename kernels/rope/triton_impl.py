import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    # locate start address
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    batch_idx = pid // sl
    cos_row_idx = pid % sl
    
    # Simple case: cos/sin batch size is 1
    cos_row_ptr = cos_ptr + cos_row_idx * cos_row_stride
    sin_row_ptr = sin_ptr + cos_row_idx * sin_row_stride

    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos_row_ptr + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin_row_ptr + cos_offsets, mask=cos_mask, other=0)

    # left half of the head
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0)

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=first_q_mask, other=0)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=first_k_mask, other=0)

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)


def run(*, q, k, cos, sin, grad_output) -> tuple:
    """Triton RoPE wrapper."""
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # In Triton kernels, we usually want (bsz, seq_len, num_heads, head_dim) for better access patterns
    # if it matches physical storage. Liger-Kernel transposes them.
    
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_rope[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    return q.transpose(1, 2), k.transpose(1, 2)


def run_backward(*, q, k, cos, sin, grad_output) -> tuple:
    """Triton RoPE backward wrapper."""
    dq_embed, dk_embed = grad_output
    # RoPE backward is RoPE with -sin
    return run(q=dq_embed, k=dk_embed, cos=cos, sin=-sin, grad_output=None)

