import torch
import triton
import triton.language as tl

@triton.jit
def mm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0).to(tl.int64)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    A_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_k = (k + rk) < K
        a = tl.load(A_ptr + k * stride_ak, mask=mask_k[None, :], other=0.0)
        b = tl.load(B_ptr + k * stride_bk, mask=mask_k[:, None], other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_ptr, acc.to(C.dtype.element_ty), mask=mask)


def run(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    GROUP_M = 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=4, num_stages=2
    )
    return c
