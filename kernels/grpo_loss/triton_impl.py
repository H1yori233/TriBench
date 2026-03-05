"""
GRPO loss Triton implementation (from Liger-Kernel).
Only GRPO loss type (PPO clipping); no KL (beta=0), no vLLM IS.
"""
import torch
import triton
import triton.language as tl

_LOSS_TYPE_GRPO = 0


@triton.jit
def _grpo_loss_fwd_kernel(
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    COMPLETION_MASK,
    ADVANTAGES,
    VLLM_IS_RATIO,
    VLLM_IS_RATIO_STRIDE,
    LOSS,
    LSE,
    KL,
    IS_CLIPPED,
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW,
    EPS_HIGH,
    LOSS_TYPE: tl.constexpr,
    SAPO_TEMP_POS,
    SAPO_TEMP_NEG,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    COMPLETION_MASK += off_b * L + off_l
    not_skip = tl.load(COMPLETION_MASK)
    if not_skip == 0:
        return

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LOSS += off_b * L + off_l
    LSE += off_b * L + off_l
    IS_CLIPPED += off_b * L + off_l

    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse
    OLD_LOGP += off_b * L + off_l
    old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    advantage = tl.load(ADVANTAGES).to(tl.float32)

    if LOSS_TYPE == 0:  # GRPO: standard PPO clipping
        coef_2 = tl.clamp(coef_1, 1.0 - EPS_LOW, 1.0 + EPS_HIGH)
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)
        is_low_clipped = (coef_1 < 1.0 - EPS_LOW) & (advantage < 0)
        is_high_clipped = (coef_1 > 1.0 + EPS_HIGH) & (advantage > 0)
        is_clipped = is_low_clipped | is_high_clipped
    else:
        per_token_loss = 0.0
        is_clipped = 0.0

    vllm_is_ratio = tl.load(
        VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE
    ).to(tl.float32)
    per_token_loss = per_token_loss * vllm_is_ratio

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
        per_token_loss += BETA * kl
        tl.store(KL, kl)

    tl.store(LOSS, per_token_loss)
    tl.store(LSE, lse)
    tl.store(IS_CLIPPED, is_clipped)


@triton.jit
def _grpo_loss_bwd_kernel(
    DLOSS,
    DLOGITS,
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    ADVANTAGES,
    COMPLETION_MASK,
    LSE,
    VLLM_IS_RATIO,
    VLLM_IS_RATIO_STRIDE,
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW,
    EPS_HIGH,
    LOSS_TYPE: tl.constexpr,
    SAPO_TEMP_POS,
    SAPO_TEMP_NEG,
    loss_stride0,
    loss_stride1,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    DLOGITS += off_b * (L + 1) * N + off_l * N
    COMPLETION_MASK += off_b * L + off_l
    not_skip = tl.load(COMPLETION_MASK)
    if not_skip == 0:
        for start in range(0, N, BLOCK_N):
            cols = tl.arange(0, BLOCK_N) + start
            tl.store(DLOGITS + cols, 0.0, mask=cols < N)
        return

    LOGITS += off_b * (L + 1) * N + off_l * N
    DLOSS += off_b * loss_stride0 + off_l * loss_stride1
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LSE += off_b * L + off_l

    dloss = tl.load(DLOSS).to(tl.float32)
    lse = tl.load(LSE).to(tl.float32)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse
    OLD_LOGP += off_b * L + off_l
    old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    advantage = tl.load(ADVANTAGES).to(tl.float32)

    if LOSS_TYPE == 0:  # GRPO
        coef_2 = tl.clamp(coef_1, 1.0 - EPS_LOW, 1.0 + EPS_HIGH)
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        mask = per_token_loss2 >= per_token_loss1
        dlogp = -per_token_loss1 * mask
    else:
        dlogp = 0.0

    vllm_is_ratio = tl.load(
        VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE
    ).to(tl.float32)
    dlogp = dlogp * vllm_is_ratio

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        dlogp += BETA * (1.0 - tl.exp(ref_logp - logp))

    dlogp = dlogp * dloss / TEMPERATURE
    tl.debug_barrier()
    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
        probs = tl.exp(logits - lse)
        dlogits = tl.where(cols == idx, 1.0 - probs, -probs) * dlogp
        tl.store(DLOGITS + cols, dlogits, mask=cols < N)


def _run_forward(
    logits,
    old_logp,
    ref_logp_dummy,
    completion_ids,
    completion_mask,
    advantages,
    vllm_is_ratio,
    temperature,
    beta,
    eps_low,
    eps_high,
):
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    lse = torch.zeros_like(loss)
    is_clipped = torch.zeros_like(loss)
    kl = torch.zeros_like(loss) if beta != 0.0 else None
    BLOCK_N = min(2048, triton.next_power_of_2(N))
    vllm_stride = vllm_is_ratio.shape[1] if vllm_is_ratio.dim() > 1 else 1
    _grpo_loss_fwd_kernel[(B, L)](
        logits,
        old_logp,
        ref_logp_dummy,
        completion_ids,
        completion_mask,
        advantages,
        vllm_is_ratio,
        vllm_stride,
        loss,
        lse,
        kl if beta != 0.0 else loss,  # dummy store
        is_clipped,
        temperature,
        beta,
        eps_low,
        eps_high,
        _LOSS_TYPE_GRPO,
        1.0,
        1.05,
        L=L,
        N=N,
        BLOCK_N=BLOCK_N,
        num_stages=2,
        num_warps=1,
    )
    return loss, lse


def run(
    *,
    logits: torch.Tensor,
    old_logp: torch.Tensor,
    completion_ids: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    temperature: float,
    eps_low: float,
    eps_high: float,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    ref_logp_dummy = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    vllm_is_ratio = torch.ones(B, L, device=logits.device, dtype=torch.float32)
    loss, _ = _run_forward(
        logits,
        old_logp,
        ref_logp_dummy,
        completion_ids,
        completion_mask,
        advantages,
        vllm_is_ratio,
        temperature,
        0.0,
        eps_low,
        eps_high,
    )
    return loss


def run_backward(
    *,
    logits: torch.Tensor,
    old_logp: torch.Tensor,
    completion_ids: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    temperature: float,
    eps_low: float,
    eps_high: float,
    grad_output: torch.Tensor,
):
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    ref_logp_dummy = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    vllm_is_ratio = torch.ones(B, L, device=logits.device, dtype=torch.float32)
    _, lse = _run_forward(
        logits,
        old_logp,
        ref_logp_dummy,
        completion_ids,
        completion_mask,
        advantages,
        vllm_is_ratio,
        temperature,
        0.0,
        eps_low,
        eps_high,
    )
    dlogits = torch.empty_like(logits)
    BLOCK_N = min(4096, triton.next_power_of_2(N))
    vllm_stride = vllm_is_ratio.shape[1]
    _grpo_loss_bwd_kernel[(B, L)](
        grad_output,
        dlogits,
        logits,
        old_logp,
        ref_logp_dummy,
        completion_ids,
        advantages,
        completion_mask,
        lse,
        vllm_is_ratio,
        vllm_stride,
        temperature,
        0.0,
        eps_low,
        eps_high,
        _LOSS_TYPE_GRPO,
        1.0,
        1.05,
        grad_output.stride(0),
        grad_output.stride(1),
        L=L,
        N=N,
        BLOCK_N=BLOCK_N,
        num_stages=1,
        num_warps=16,
    )
    dlogits[:, -1, :] = 0
    return dlogits
