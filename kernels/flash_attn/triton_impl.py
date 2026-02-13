import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # -- grid id --
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(SEQ_LEN, BLOCK_SIZE_Q)
    num_pid_n = tl.cdiv(SEQ_LEN, BLOCK_SIZE_K)
    
    pid_batch_head = tl.program_id(1)
    batch_idx = pid_batch_head // NUM_HEADS
    head_idx = pid_batch_head % NUM_HEADS
    
    # -- pointers --
    Q += batch_idx * stride_qb + head_idx * stride_qh
    K += batch_idx * stride_kb + head_idx * stride_kh
    V += batch_idx * stride_vb + head_idx * stride_vh
    Out += batch_idx * stride_ob + head_idx * stride_oh
    
    # -- loop over q blocks --
    offs_m = pid * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_n = tl.arange(0, BLOCK_SIZE_K)
    offs_k = tl.arange(0, HEAD_DIM)
    
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    
    # -- load Q --
    q = tl.load(q_ptrs)
    
    # -- initialize --
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    
    # -- loop over k blocks --
    lo = 0
    hi = SEQ_LEN
    if IS_CAUSAL:
        hi = (pid + 1) * BLOCK_SIZE_Q
    
    for start_n in range(lo, hi, BLOCK_SIZE_K):
        # -- load K, V --
        k_ptrs = K + (offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk) + start_n * stride_kn
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk) + start_n * stride_vn
        
        k = tl.load(k_ptrs) # (D, K)
        
        # -- compute qk --
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
            
        # -- softmax --
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # -- update acc --
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)
        
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptrs) # (K, D)
        p = p * beta[:, None]
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        
        l_i = l_i * alpha + l_ij * beta
        m_i = m_next
        
    # -- finalize --
    acc = acc / l_i[:, None]
    
    # -- store --
    off_o = offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(Out + off_o, acc.to(Out.dtype.element_ty))

def run_flash_attn(q, k, v, causal=False, sm_scale=None):
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    
    batch, nheads, seqlen, hdim = q.shape
    out = torch.empty_like(q)
    
    grid = (triton.cdiv(seqlen, 128), batch * nheads) # BLOCK_SIZE_Q will be autotuned
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, nheads, seqlen, hdim,
        IS_CAUSAL=causal,
    )
    return out
