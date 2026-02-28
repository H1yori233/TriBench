import torch
import triton
import triton.language as tl

@triton.jit
def baddbmm_kernel(
    A, B, O, bias,
    alpha, beta,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_om, stride_on,
    stride_biasb, stride_biasm, stride_biasn,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # batch offset
    pid_b = tl.program_id(2).to(tl.int64)
    
    A_batch = A + pid_b * stride_ab
    B_batch = B + pid_b * stride_bb
    O_batch = O + pid_b * stride_ob
    bias_batch = bias + pid_b * stride_biasb

    pidx = tl.program_id(0).to(tl.int64)
    pidy = tl.program_id(1).to(tl.int64)

    gridx = tl.num_programs(0)
    gridy = tl.num_programs(1)
    
    # Swizzling for better L2 cache hit
    pid = pidx + pidy * gridx
    num_CTA_per_group = gridy * GROUP_M
    group_id = pid // num_CTA_per_group
    inner_group_id = pid % num_CTA_per_group
    GROUP_SIZE = tl.where((group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
    pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_batch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, K, TILE_K):
        mask_k = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    bias_ptrs = bias_batch + offs_m[:, None] * stride_biasm + offs_n[None, :] * stride_biasn
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_c = mask_m[:, None] & mask_n[None, :]

    bi = tl.load(bias_ptrs, mask=mask_c, other=0.0)
    out = accumulator * alpha + bi * beta
    
    o_ptrs = O_batch + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=mask_c)


def run(bias, A, B, beta=1.0, alpha=1.0, grad_output=None):
    batch, M, K = A.shape
    _, _, N = B.shape
    
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    # Simplified fixed config (heuristic can be added later)
    TILE_M = 64
    TILE_N = 64
    TILE_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N), batch)
    
    baddbmm_kernel[grid](
        A, B, out, bias,
        alpha, beta,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        bias.stride(0), bias.stride(1), bias.stride(2),
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
        GROUP_M=GROUP_M,
        num_warps=4, num_stages=3
    )
    return out



def run_backward(bias, A, B, beta=1.0, alpha=1.0, grad_output=None):
    # dA = alpha * (grad_output @ B.T)
    # dB = alpha * (A.T @ grad_output)
    # dbias = beta * grad_output
    
    batch, M, K = A.shape
    _, _, N = B.shape
    
    # We reuse 'run' to compute dA and dB by passing appropriate inputs
    # run(bias, A, B, beta, alpha) -> beta*bias + alpha * (A@B)
    
    # 1. dA = alpha * (grad_output @ B.T)
    #    B.T is (batch, N, K)
    #    grad_output is (batch, M, N)
    #    dA is (batch, M, K)
    zero_bias_K = torch.zeros((batch, M, K), dtype=A.dtype, device=A.device)
    dA = run(zero_bias_K, grad_output, B.transpose(1, 2), beta=0.0, alpha=alpha)
    
    # 2. dB = alpha * (A.T @ grad_output)
    #    A.T is (batch, K, M)
    #    grad_output is (batch, M, N)
    #    dB is (batch, K, N)
    zero_bias_N = torch.zeros((batch, K, N), dtype=B.dtype, device=B.device)
    dB = run(zero_bias_N, A.transpose(1, 2), grad_output, beta=0.0, alpha=alpha)
    
    # 3. dbias = beta * grad_output
    dbias = beta * grad_output
    
    return dbias, dA, dB

