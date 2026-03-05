"""
Tiled MLP: chunk x along dim 0, run linear per shard, concat.
No Triton kernel - pure PyTorch tiling for memory efficiency (same math as reference).
"""
import torch


def run(*, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, shards: int, grad_output: torch.Tensor) -> torch.Tensor:
    """Tiled linear: chunk x, y_i = x_i @ W.t() + b, concat."""
    x_shards = torch.chunk(x, chunks=shards, dim=0)
    out_shards = [xi @ W.t() + b for xi in x_shards]
    return torch.cat(out_shards, dim=0)


def run_backward(*, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, shards: int, grad_output: torch.Tensor):
    """Tiled backward: chunk grad_output and x, accumulate W.grad and b.grad."""
    x_grad = torch.empty_like(x)
    W_grad = torch.zeros_like(W)
    b_grad = torch.zeros_like(b)
    go_shards = torch.chunk(grad_output, chunks=shards, dim=0)
    x_shards = torch.chunk(x, chunks=shards, dim=0)
    start = 0
    for go_i, x_i in zip(go_shards, x_shards):
        n = go_i.shape[0]
        x_grad[start:start + n] = go_i @ W
        W_grad.add_(go_i.t() @ x_i)
        b_grad.add_(go_i.sum(0))
        start += n
    return x_grad, W_grad, b_grad
