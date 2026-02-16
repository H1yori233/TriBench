import torch
import triton
import triton.language as tl

@triton.jit
def _jsd_kernel(
    X_ptr,  # input in logspace, X = log Q
    X_stride,
    Y_ptr,  # ground truth in logspace, Y = log P
    Y_stride,
    loss_ptr,
    loss_stride,
    beta: tl.constexpr,
    n_non_ignore: int,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        if beta == 0.0:  # forward KL
            Y_prob = tl.exp(Y)
            loss = Y_prob * (Y - X)
        elif beta == 1.0:  # reverse KL
            X_prob = tl.exp(X)
            loss = X_prob * (X - Y)
        else:
            P = tl.exp(Y)
            Q = tl.exp(X)
            M = beta * P + (1 - beta) * Q
            log_M = tl.log(M)
            loss = beta * P * Y + (1 - beta) * Q * X - M * log_M

        scale = 1.0 / n_non_ignore
        loss = loss * scale
        tl.store(loss_ptr + offsets, loss, mask=mask)


def run(*, _input: torch.Tensor, target: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Triton JSD wrapper."""
    BT, V = _input.shape
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    
    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    n_non_ignore = BT

    _jsd_kernel[(BT,)](
        X_ptr=_input,
        X_stride=_input.stride(0),
        Y_ptr=target,
        Y_stride=target.stride(0),
        loss_ptr=loss,
        loss_stride=loss.stride(0),
        beta=beta,
        n_non_ignore=n_non_ignore,
        n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    return torch.sum(loss).to(_input.dtype)
