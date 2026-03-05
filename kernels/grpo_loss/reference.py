"""
Reference implementation for GRPO loss (PPO-style clipping).
Matches the Triton kernel logic for loss_type=grpo (no KL, no vLLM IS).
"""
import torch
import torch.nn.functional as F


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 2)
    L = params.get("L", 8)
    N = params.get("N", 256)
    temperature = params.get("temperature", 0.9)
    eps_low = params.get("eps_low", 0.2)
    eps_high = params.get("eps_high", 0.2)

    # logits: (B, L+1, N) - last position not used for completion logp in kernel
    logits = torch.randn((B, L + 1, N), dtype=dtype, device=device, requires_grad=True)
    completion_ids = torch.randint(0, N, (B, L), device=device, dtype=torch.long)
    old_logp = torch.randn((B, L), dtype=torch.float32, device=device)
    advantages = torch.randn((B,), dtype=torch.float32, device=device)
    completion_mask = torch.ones((B, L), dtype=torch.float32, device=device)
    grad_output = torch.ones((B, L), dtype=torch.float32, device=device)
    return {
        "logits": logits,
        "old_logp": old_logp,
        "completion_ids": completion_ids,
        "advantages": advantages,
        "completion_mask": completion_mask,
        "temperature": temperature,
        "eps_low": eps_low,
        "eps_high": eps_high,
        "grad_output": grad_output,
    }


def ref(
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
    """GRPO per-token loss (B, L). Uses positions 0..L-1 of logits."""
    B, L_plus_1, N = logits.shape
    L = L_plus_1 - 1
    logits_slice = logits[:, :L, :].float() / temperature  # (B, L, N)
    log_probs = F.log_softmax(logits_slice, dim=-1)  # (B, L, N)
    # Gather logp at completion_ids
    logp = torch.gather(log_probs, 2, completion_ids.unsqueeze(2)).squeeze(2)  # (B, L)
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    adv = advantages.unsqueeze(1).expand(B, L)
    loss1 = ratio * adv
    loss2 = clipped * adv
    per_token_loss = -torch.minimum(loss1, loss2)
    return (per_token_loss * completion_mask).to(torch.float32)


def ref_backward(
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
    logits = logits.detach().requires_grad_(True)
    loss = ref(
        logits=logits,
        old_logp=old_logp,
        completion_ids=completion_ids,
        advantages=advantages,
        completion_mask=completion_mask,
        temperature=temperature,
        eps_low=eps_low,
        eps_high=eps_high,
        grad_output=grad_output,
    )
    loss.backward(grad_output)
    return logits.grad
