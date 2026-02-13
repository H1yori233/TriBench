import torch


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 1)
    T = params.get("T", 4096)
    N_Q_H = params.get("N_Q_H", 32)
    N_KV_H = params.get("N_KV_H", 8)
    H = params.get("H", 128)

    q = torch.randn((B, N_Q_H, T, H), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((B, N_KV_H, T, H), dtype=dtype, device=device, requires_grad=True)
    
    # RoPE cos/sin usually have H//2 unique values cloned to H
    cos_half = torch.randn((1, T, H // 2), dtype=dtype, device=device)
    sin_half = torch.randn((1, T, H // 2), dtype=dtype, device=device)
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)

    return {"q": q, "k": k, "cos": cos, "sin": sin}


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos, sin: [1, seq_len, head_dim]
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def ref(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Wrapper for reference implementation."""
    return apply_rotary_pos_emb(q, k, cos, sin)
