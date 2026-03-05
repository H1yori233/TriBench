"""
Group Normalization Triton implementation (from Liger-Kernel).
Stripped of liger_kernel dependencies; uses tl.math.rsqrt for CUDA.
"""
import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536


@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    Y_col_stride,
    X_ptr,
    X_row_stride,
    X_col_stride,
    Mean_ptr,
    Mean_row_stride,
    Mean_col_stride,
    RSTD_ptr,
    RSTD_row_stride,
    RSTD_col_stride,
    W_ptr,
    B_ptr,
    hidden_size,
    channels_per_group,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride

    block_range = tl.arange(0, BLOCK_SIZE)

    s = 0.0
    squared_sum = 0.0
    for i in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + block_range
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(X)
        squared_sum += tl.sum(X * X)

    m = s / hidden_size
    variance = (squared_sum / hidden_size) - (m * m)
    rstd = tl.math.rsqrt(variance + eps)

    hidden_size_per_channel = hidden_size // channels_per_group
    for i in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + block_range
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m).to(tl.float32)
        local_channel = hidden_size_offsets // hidden_size_per_channel
        global_channel = group_idx * channels_per_group + local_channel
        W = tl.load(W_ptr + global_channel, mask=mask).to(tl.float32)
        B = tl.load(B_ptr + global_channel, mask=mask).to(tl.float32)
        Y = (X - m) * rstd * W + B
        tl.store(Y_ptr + hidden_size_offsets, Y.to(Y_ptr.dtype.element_ty), mask=mask)

    tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride, m)
    tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride, rstd)


@triton.jit
def _group_norm_backward_kernel(
    X_ptr,
    X_row_stride,
    X_col_stride,
    W_ptr,
    Mean_ptr,
    Mean_ptr_row_stride,
    Mean_ptr_col_stride,
    RSTD_ptr,
    DX_ptr,
    DW_ptr,
    DB_ptr,
    UPSTREAM_ptr,
    hidden_size: tl.constexpr,
    channels_per_group: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    X_ptr += batch_idx * X_row_stride
    DX_ptr += batch_idx * X_row_stride
    UPSTREAM_ptr += batch_idx * X_row_stride

    mean = tl.load(Mean_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)
    rstd = tl.load(RSTD_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)

    c1 = 0.0
    c2 = 0.0
    block_range = tl.arange(0, BLOCK_SIZE)

    for local_i in range(channels_per_group):
        channel_idx = group_idx * channels_per_group + local_i
        dW = 0.0
        dB = 0.0
        W = tl.load(W_ptr + channel_idx)
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(
                X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )
            UPSTREAM_grad = tl.load(
                UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )
            x_hat = (X - mean) * rstd
            dW += tl.sum(UPSTREAM_grad * x_hat)
            dB += tl.sum(UPSTREAM_grad)
            wdy = W * UPSTREAM_grad
            c1 += tl.sum(x_hat * wdy)
            c2 += tl.sum(wdy)
        tl.atomic_add(DW_ptr + channel_idx, dW.to(dtype))
        tl.atomic_add(DB_ptr + channel_idx, dB.to(dtype))

    N = hidden_size * channels_per_group
    c1 = c1 / N
    c2 = c2 / N

    for local_i in range(channels_per_group):
        channel_idx = group_idx * channels_per_group + local_i
        W = tl.load(W_ptr + channel_idx)
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(
                X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )
            UPSTREAM_grad = tl.load(
                UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )
            x_hat = (X - mean) * rstd
            wdy = W * UPSTREAM_grad
            dx = (wdy - (x_hat * c1 + c2)) * rstd
            tl.store(DX_ptr + channel_idx * X_col_stride + hidden_size_offsets, dx, mask=mask)


def run(
    *,
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor,
    num_groups: int,
    eps: float,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    shape = X.shape
    batch_size = shape[0]
    num_channels = shape[1]
    channels_per_group = num_channels // num_groups
    X_flat = X.view(batch_size, num_groups, -1).contiguous()
    hidden_size = X_flat.shape[-1]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))

    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    Mean = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)
    RSTD = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)

    _group_norm_forward_kernel[(batch_size, num_groups)](
        Y,
        Y.stride(0),
        Y.stride(1),
        X_flat,
        X_flat.stride(0),
        X_flat.stride(1),
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        RSTD.stride(0),
        RSTD.stride(1),
        W,
        B,
        hidden_size=hidden_size,
        channels_per_group=channels_per_group,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return Y.view(*shape)


def run_backward(
    *,
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor,
    num_groups: int,
    eps: float,
    grad_output: torch.Tensor,
):
    shape = X.shape
    batch_size = shape[0]
    num_channels = shape[1]
    channels_per_group = num_channels // num_groups
    X_flat = X.view(batch_size, num_groups, -1).contiguous()
    dY = grad_output.view(batch_size, num_groups, -1)
    hidden_size = X_flat.shape[-1]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))

    DX = torch.empty_like(X_flat)
    DW = torch.zeros((num_channels,), dtype=W.dtype, device=W.device)
    DB = torch.zeros((num_channels,), dtype=B.dtype, device=B.device)
    triton_dtype = tl.float32 if X.dtype == torch.float32 else tl.bfloat16

    # Need mean/rstd from forward for backward
    Mean = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)
    RSTD = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)
    _group_norm_forward_kernel[(batch_size, num_groups)](
        torch.empty_like(X_flat),
        X_flat.stride(0),
        X_flat.stride(1),
        X_flat,
        X_flat.stride(0),
        X_flat.stride(1),
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        RSTD.stride(0),
        RSTD.stride(1),
        W,
        B,
        hidden_size=hidden_size,
        channels_per_group=channels_per_group,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    _group_norm_backward_kernel[(batch_size, num_groups)](
        X_flat,
        X_flat.stride(0),
        X_flat.stride(1),
        W,
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        DX,
        DW,
        DB,
        dY,
        hidden_size=hidden_size,
        channels_per_group=channels_per_group,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype,
    )
    return DX.view(*shape), DW, DB
