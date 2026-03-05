"""
Reference: Qwen2VL M-RoPE with 3 sections (t, h, w).
cos/sin shape (3, B, S, head_dim); section bounds t_end, h_end apply to the left half indices.
"""
import torch


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 2)
    S = params.get("S", 64)
    N_Q_H = params.get("N_Q_H", 8)
    N_KV_H = params.get("N_KV_H", 4)
    H = params.get("H", 128)
    t_end = params.get("mrope_section_t", 16)
    h_len = params.get("mrope_section_h", 16)
    assert t_end <= H // 2 and t_end + h_len <= H // 2

    q = torch.randn((B, N_Q_H, S, H), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((B, N_KV_H, S, H), dtype=dtype, device=device, requires_grad=True)
    cos = torch.randn((3, B, S, H), dtype=dtype, device=device)
    sin = torch.randn((3, B, S, H), dtype=dtype, device=device)
    mrope_section = (t_end, h_len)
    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    return {
        "q": q,
        "k": k,
        "cos": cos,
        "sin": sin,
        "mrope_section": mrope_section,
        "grad_output": (dq, dk),
    }


def _apply_mrope(q, k, cos, sin, mrope_section):
    """Apply M-RoPE: 3-section cos/sin, rotation on left/right half of head_dim."""
    t_end, h_section_len = mrope_section
    h_end = t_end + h_section_len
    B, N_Q_H, S, H = q.shape
    N_KV_H = k.shape[1]
    half = H // 2
    dtype = q.dtype
    device = q.device

    q_out = q.clone()
    k_out = k.clone()
    for b in range(B):
        for s in range(S):
            cos_row = torch.zeros(half, dtype=dtype, device=device)
            sin_row = torch.zeros(half, dtype=dtype, device=device)
            if t_end > 0:
                cos_row[:t_end] = cos[0, b, s, :t_end]
                sin_row[:t_end] = sin[0, b, s, :t_end]
            if h_end > t_end:
                cos_row[t_end:h_end] = cos[1, b, s, t_end:h_end]
                sin_row[t_end:h_end] = sin[1, b, s, t_end:h_end]
            if half > h_end:
                cos_row[h_end:half] = cos[2, b, s, h_end:half]
                sin_row[h_end:half] = sin[2, b, s, h_end:half]
            cos_full = torch.cat([cos_row, cos_row], dim=0)
            sin_full = torch.cat([sin_row, sin_row], dim=0)
            for h in range(N_Q_H):
                q1 = q_out[b, h, s, :half]
                q2 = q_out[b, h, s, half:]
                q_out[b, h, s, :half] = q1 * cos_row - q2 * sin_row
                q_out[b, h, s, half:] = q2 * cos_row + q1 * sin_row
            for h in range(N_KV_H):
                k1 = k_out[b, h, s, :half]
                k2 = k_out[b, h, s, half:]
                k_out[b, h, s, :half] = k1 * cos_row - k2 * sin_row
                k_out[b, h, s, half:] = k2 * cos_row + k1 * sin_row
    return q_out, k_out


def ref(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: tuple,
    grad_output: tuple,
) -> tuple:
    return _apply_mrope(q, k, cos, sin, mrope_section)


def ref_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: tuple,
    grad_output: tuple,
):
    q_emb, k_emb = _apply_mrope(q, k, cos, sin, mrope_section)
    dq, dk = grad_output
    q.grad = None
    k.grad = None
    torch.autograd.backward([q_emb, k_emb], [dq, dk])
    return q.grad, k.grad
