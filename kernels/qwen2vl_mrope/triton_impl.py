"""
Qwen2VL M-RoPE Triton kernel (from temp_for_reference/qwen2vl_mrope.py).
Forward/backward by same kernel with BACKWARD_PASS flag; we clone q,k so we don't mutate inputs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_qwen2vl_mrope(
    q_ptr,
    k_ptr,
    cos,
    sin,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)
    t_end = mrope_section_t
    h_end = t_end + mrope_section_h

    t_cos = cos + pid * hd
    h_cos = t_cos + bs * sl * hd
    w_cos = h_cos + bs * sl * hd
    t_sin = sin + pid * hd
    h_sin = t_sin + bs * sl * hd
    w_sin = h_sin + bs * sl * hd

    cos_offsets = tl.arange(0, pad_hd // 2)
    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    w_mask = (h_end <= cos_offsets) & (cos_offsets < hd // 2)
    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    if not BACKWARD_PASS:
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def _mrope_forward(q, k, cos, sin, mrope_section):
    q = q.transpose(1, 2).contiguous().clone()
    k = k.transpose(1, 2).contiguous().clone()
    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)
    n_row = batch_size * seq_len
    # Kernel expects cos/sin as (3, bs*sl, hd) so cos + pid*hd indexes position pid in section 0
    cos = cos.reshape(3, batch_size * seq_len, head_dim).contiguous()
    sin = sin.reshape(3, batch_size * seq_len, head_dim).contiguous()
    t_end, h_len = mrope_section
    mrope_section_h = h_len
    _triton_qwen2vl_mrope[(n_row,)](
        q, k, cos, sin, seq_len, batch_size, n_q_head, n_kv_head, head_dim,
        pad_n_q_head, pad_n_kv_head, pad_hd, t_end, mrope_section_h,
        BLOCK_SIZE=BLOCK_SIZE, BACKWARD_PASS=False,
    )
    return q.transpose(1, 2), k.transpose(1, 2)


def _mrope_backward(dq, dk, cos, sin, mrope_section):
    dq = dq.transpose(1, 2).contiguous().clone()
    dk = dk.transpose(1, 2).contiguous().clone()
    batch_size, seq_len, n_q_head, head_dim = dq.shape
    n_kv_head = dk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)
    n_row = batch_size * seq_len
    cos = cos.reshape(3, batch_size * seq_len, head_dim).contiguous()
    sin = sin.reshape(3, batch_size * seq_len, head_dim).contiguous()
    t_end, h_len = mrope_section
    mrope_section_h = h_len
    _triton_qwen2vl_mrope[(n_row,)](
        dq, dk, cos, sin, seq_len, batch_size, n_q_head, n_kv_head, head_dim,
        pad_n_q_head, pad_n_kv_head, pad_hd, t_end, mrope_section_h,
        BLOCK_SIZE=BLOCK_SIZE, BACKWARD_PASS=True,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2)


def run(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: tuple,
    grad_output: tuple,
) -> tuple:
    q_out, k_out = _mrope_forward(q.clone(), k.clone(), cos, sin, mrope_section)
    return (q_out, k_out)


def run_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: tuple,
    grad_output: tuple,
):
    dq, dk = grad_output
    return _mrope_backward(dq, dk, cos, sin, mrope_section)
