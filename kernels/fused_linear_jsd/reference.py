"""
Reference: Linear (student + teacher) + JSD in PyTorch.
student_logits = student_input @ student_weight.t(), same for teacher;
JSD between log_softmax(student_logits/T) and log_softmax(teacher_logits/T).
"""
import torch


def make_inputs(params: dict, device: str, seed: int, dtype: torch.dtype):
    torch.manual_seed(seed)
    B = params.get("B", 2)
    T = params.get("T", 128)
    H = params.get("H", 512)
    V = params.get("V", 2048)
    BT = B * T
    student_input = torch.randn((BT, H), dtype=dtype, device=device, requires_grad=True)
    student_weight = torch.randn((V, H), dtype=dtype, device=device, requires_grad=True)
    teacher_input = torch.randn((BT, H), dtype=dtype, device=device, requires_grad=True)
    teacher_weight = torch.randn((V, H), dtype=dtype, device=device, requires_grad=True)
    jsd_beta = params.get("jsd_beta", 0.5)
    temperature = params.get("temperature", 1.0)
    grad_output = torch.tensor(1.0, dtype=torch.float32, device=device)
    return {
        "student_input": student_input,
        "student_weight": student_weight,
        "teacher_input": teacher_input,
        "teacher_weight": teacher_weight,
        "jsd_beta": jsd_beta,
        "temperature": temperature,
        "grad_output": grad_output,
    }


def _jsd_loss(student_log: torch.Tensor, teacher_log: torch.Tensor, beta: float) -> torch.Tensor:
    """JSD in log space; return scalar (mean over rows)."""
    P = torch.exp(teacher_log)
    Q = torch.exp(student_log)
    M = (beta * P + (1 - beta) * Q).clamp(min=1e-12)
    log_M = torch.log(M)
    kl_pm = (P * (teacher_log - log_M)).sum(dim=-1)
    kl_qm = (Q * (student_log - log_M)).sum(dim=-1)
    loss_per_row = beta * kl_pm + (1 - beta) * kl_qm
    return loss_per_row.mean()


def ref(
    *,
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    jsd_beta: float,
    temperature: float,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    student_logits = (student_input @ student_weight.t()).float() / temperature
    teacher_logits = (teacher_input @ teacher_weight.t()).float() / temperature
    student_log = torch.log_softmax(student_logits, dim=-1)
    teacher_log = torch.log_softmax(teacher_logits, dim=-1)
    return _jsd_loss(student_log, teacher_log, jsd_beta)


def ref_backward(
    *,
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    jsd_beta: float,
    temperature: float,
    grad_output: torch.Tensor,
):
    student_input = student_input.detach().requires_grad_(True)
    student_weight = student_weight.detach().requires_grad_(True)
    loss = ref(
        student_input=student_input,
        student_weight=student_weight,
        teacher_input=teacher_input,
        teacher_weight=teacher_weight,
        jsd_beta=jsd_beta,
        temperature=temperature,
        grad_output=grad_output,
    )
    loss.backward(grad_output)
    return student_input.grad, student_weight.grad
