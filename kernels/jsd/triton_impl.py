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


@triton.jit
def _jsd_kernel_backward(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    dX_ptr,
    dX_stride,
    dY_ptr,
    dY_stride,
    grad_output_ptr,
    beta: tl.constexpr,
    n_non_ignore: int,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    Y_ptr += pid * Y_stride
    dX_ptr += pid * dX_stride
    dY_ptr += pid * dY_stride

    # grad_output is typically a scalar for JSD (since output is a scalar loss)
    grad_output = tl.load(grad_output_ptr)
    scale = grad_output / n_non_ignore

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        P = tl.exp(Y)
        Q = tl.exp(X)
        M = beta * P + (1.0 - beta) * Q
        log_M = tl.log(M)

        # dL/dX = Q * (1-beta) * (X - log_M)
        dX_row = Q * (1.0 - beta) * (X - log_M)
        # dL/dY = P * beta * (Y - log_M)
        dY_row = P * beta * (Y - log_M)

        tl.store(dX_ptr + offsets, (dX_row * scale).to(dX_ptr.dtype.element_ty), mask=mask)
        tl.store(dY_ptr + offsets, (dY_row * scale).to(dY_ptr.dtype.element_ty), mask=mask)



def run(*, _input: torch.Tensor, target: torch.Tensor, beta: float = 0.5, grad_output: torch.Tensor) -> torch.Tensor:
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


def run_backward(*, _input: torch.Tensor, target: torch.Tensor, beta: float = 0.5, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton JSD backward wrapper."""
    BT, V = _input.shape
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    
    dX = torch.empty_like(_input)
    dY = torch.empty_like(target)
    n_non_ignore = BT

    _jsd_kernel_backward[(BT,)](
        X_ptr=_input,
        X_stride=_input.stride(0),
        Y_ptr=target,
        Y_stride=target.stride(0),
        dX_ptr=dX,
        dX_stride=dX.stride(0),
        dY_ptr=dY,
        dY_stride=dY.stride(0),
        grad_output_ptr=grad_output,
        beta=beta,
        n_non_ignore=n_non_ignore,
        n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    return dX, dY

