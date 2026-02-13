import torch
import torch.nn.functional as F

def make_inputs(case: dict, device: str, seed: int, dtype: torch.dtype, **kwargs):
    batch_size = case.get("batch_size", 8)
    num_heads = case.get("num_heads", 16)
    seq_len = case.get("seq_len", 4096)
    head_dim = case.get("head_dim", 64)
    
    q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    return {"q": q, "k": k, "v": v, "causal": False}

def ref(q, k, v, causal=False, sm_scale=None):
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        seq_len = q.shape[2]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=q.device)) == 0
        p = p.masked_fill(mask, float("-inf"))
    attn = F.softmax(p, dim=-1)
    return torch.matmul(attn, v)

def estimate(case: dict):
    batch_size = case.get("batch_size", 8)
    num_heads = case.get("num_heads", 16)
    seq_len = case.get("seq_len", 4096)
    head_dim = case.get("head_dim", 64)
    
    flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    bytes_io = 4 * batch_size * num_heads * seq_len * head_dim * 2
    return {"flops": flops, "bytes": bytes_io}
