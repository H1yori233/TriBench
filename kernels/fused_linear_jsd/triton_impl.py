"""
Fused Linear + JSD: chunked forward using JSD Triton kernel.
Student/teacher logits from linear, log_softmax with temperature, then JSD loss.
No label/ignore_index (has_label=False). Backward recomputes and scales by grad_output.
"""
import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def _jsd_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
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
        if beta == 0.0:
            Y_prob = tl.exp(Y)
            loss = Y_prob * (Y - X)
        elif beta == 1.0:
            X_prob = tl.exp(X)
            loss = X_prob * (X - Y)
        else:
            P = tl.exp(Y)
            Q = tl.exp(X)
            M = beta * P + (1 - beta) * Q
            M = tl.maximum(M, 1e-12)
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
        M = tl.maximum(M, 1e-12)
        log_M = tl.log(M)
        dX_row = Q * (1.0 - beta) * (X - log_M)
        dY_row = P * beta * (Y - log_M)
        tl.store(dX_ptr + offsets, (dX_row * scale).to(dX_ptr.dtype.element_ty), mask=mask)
        tl.store(dY_ptr + offsets, (dY_row * scale).to(dY_ptr.dtype.element_ty), mask=mask)


def run(
    *,
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    jsd_beta: float,
    temperature: float,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    BT, H = student_input.shape
    V = student_weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    inc_factor = (V + H - 1) // H
    chunk_size = triton.next_power_of_2((BT + inc_factor - 1) // inc_factor)
    num_chunks = (BT + chunk_size - 1) // chunk_size
    n_non_ignore = BT
    loss_acc = 0.0
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, BT)
        s_chunk = student_input[start:end]
        t_chunk = teacher_input[start:end]
        student_logits = (s_chunk @ student_weight.t()).float() / temperature
        teacher_logits = (t_chunk @ teacher_weight.t()).float() / temperature
        student_log = torch.log_softmax(student_logits, dim=-1).contiguous()
        teacher_log = torch.log_softmax(teacher_logits, dim=-1).contiguous()
        chunk_n = student_log.shape[0]
        loss_1d = torch.zeros((chunk_n, V), dtype=torch.float32, device=student_input.device)
        _jsd_kernel[(chunk_n,)](
            X_ptr=student_log,
            X_stride=student_log.stride(0),
            Y_ptr=teacher_log,
            Y_stride=teacher_log.stride(0),
            loss_ptr=loss_1d,
            loss_stride=loss_1d.stride(0),
            beta=jsd_beta,
            n_non_ignore=n_non_ignore,
            n_cols=V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
        loss_acc += loss_1d.sum().item()
    return torch.tensor(loss_acc, dtype=student_input.dtype, device=student_input.device)


def run_backward(
    *,
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    jsd_beta: float,
    temperature: float,
    grad_output: torch.Tensor,
):
    BT, H = student_input.shape
    V = student_weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    inc_factor = (V + H - 1) // H
    chunk_size = triton.next_power_of_2((BT + inc_factor - 1) // inc_factor)
    num_chunks = (BT + chunk_size - 1) // chunk_size
    n_non_ignore = BT
    dtype = student_input.dtype
    device = student_input.device
    grad_input = torch.zeros_like(student_input)
    grad_weight = torch.zeros_like(student_weight)
    go_scalar = grad_output.float().item() if grad_output.numel() == 1 else grad_output.sum().item()
    go_t = torch.tensor(go_scalar, dtype=torch.float32, device=device)
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, BT)
        s_chunk = student_input[start:end]
        t_chunk = teacher_input[start:end]
        student_logits = (s_chunk @ student_weight.t()).float() / temperature
        teacher_logits = (t_chunk @ teacher_weight.t()).float() / temperature
        student_log = torch.log_softmax(student_logits, dim=-1).contiguous()
        teacher_log = torch.log_softmax(teacher_logits, dim=-1).contiguous()
        chunk_n = student_log.shape[0]
        d_student_log = torch.empty_like(student_log)
        d_teacher_log = torch.empty_like(teacher_log)
        _jsd_kernel_backward[(chunk_n,)](
            X_ptr=student_log,
            X_stride=student_log.stride(0),
            Y_ptr=teacher_log,
            Y_stride=teacher_log.stride(0),
            dX_ptr=d_student_log,
            dX_stride=d_student_log.stride(0),
            dY_ptr=d_teacher_log,
            dY_stride=d_teacher_log.stride(0),
            grad_output_ptr=go_t,
            beta=jsd_beta,
            n_non_ignore=n_non_ignore,
            n_cols=V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
        student_probs = torch.softmax(student_logits, dim=-1)
        d_log_probs = d_student_log.float()
        d_logits = student_probs * (d_log_probs - (d_log_probs * student_probs).sum(dim=-1, keepdim=True)) / temperature
        d_logits = d_logits.to(dtype)
        grad_input[start:end] = d_logits @ student_weight
        grad_weight.add_(d_logits.t() @ s_chunk)
    return grad_input, grad_weight
