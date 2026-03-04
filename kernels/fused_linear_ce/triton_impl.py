import torch
import triton
import triton.language as tl

@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride

    if y == ignore_index:
        return

    loss_ptr += program_id * loss_stride

    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)
    loss = lse - ori_X_y
    if reduction == "mean":
        loss = loss / n_non_ignore
    tl.store(loss_ptr, loss)


@triton.jit
def liger_cross_entropy_backward_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    dX_ptr,
    dX_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride
    dX_ptr += program_id * dX_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = X_offsets < n_cols
            tl.store(dX_ptr + X_offsets, 0.0, mask=mask)
        return

    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = X_offsets < n_cols
        X_block = tl.load(X_ptr + X_offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        
        # grad = exp(x - lse)
        grad = tl.exp(X_block - lse)
        
        # if x == y, grad -= 1
        grad -= tl.where(X_offsets == y, 1.0, 0.0)
        
        if reduction == "mean":
            grad = grad / n_non_ignore
            
        tl.store(dX_ptr + X_offsets, grad.to(dX_ptr.dtype.element_ty), mask=mask)



def run(*, X: torch.Tensor, W: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, grad_output: torch.Tensor) -> torch.Tensor:
    """Triton Fused Linear + CrossEntropy (Chunked forward)."""
    BT, H = X.shape
    V = W.shape[0]
    
    # Block size and chunking logic
    MAX_FUSED_SIZE = 65536 // 2
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    
    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=X.device)
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _X_chunk = X[start_idx:end_idx]
        logits_chunk = _X_chunk @ W.t()
        target_chunk = target[start_idx:end_idx]
        
        n_rows = logits_chunk.shape[0]
        loss_1d_slice = loss_1d[start_idx:end_idx]

        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(0),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(0),
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(0),
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            label_smoothing=0.0,
            reduction="mean",
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

    return torch.sum(loss_1d).to(X.dtype)


def run_backward(*, X: torch.Tensor, W: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton Fused Linear + CrossEntropy (Chunked backward)."""
    BT, H = X.shape
    V = W.shape[0]
    
    # Block size and chunking logic
    MAX_FUSED_SIZE = 65536 // 2
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    
    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    dX = torch.zeros_like(X)
    dW = torch.zeros_like(W)
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _X_chunk = X[start_idx:end_idx]
        logits_chunk = _X_chunk @ W.t()
        target_chunk = target[start_idx:end_idx]
        
        n_rows = logits_chunk.shape[0]
        # Temporary storage for gradients wrt logits
        d_logits_chunk = torch.empty_like(logits_chunk)

        liger_cross_entropy_backward_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(0),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(0),
            dX_ptr=d_logits_chunk,
            dX_stride=d_logits_chunk.stride(0),
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            label_smoothing=0.0,
            reduction="mean",
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
        # Apply grad_output (scalar since loss is sum/mean)
        d_logits_chunk *= grad_output

        # Accumulate gradients
        # dX = dZ @ W -> (end_idx-start_idx, V) @ (V, H) = (end_idx-start_idx, H)
        dX[start_idx:end_idx] = d_logits_chunk @ W
        # dW += dZ.T @ X -> (V, end_idx-start_idx) @ (end_idx-start_idx, H) = (V, H)
        dW += d_logits_chunk.t() @ _X_chunk

    return dX, dW

