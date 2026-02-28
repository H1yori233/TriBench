import torch
import torch.nn.functional as F

def make_inputs(params, device, seed, dtype):
    torch.manual_seed(seed)
    N = params["N"]
    torch_dtype = torch.float16 if dtype == "fp16" else (torch.bfloat16 if dtype == "bf16" else torch.float32)
        
    x = torch.randn(N, device=device, dtype=torch_dtype)
    y = torch.randn(N, device=device, dtype=torch_dtype)
    return {"x": x, "y": y}

def ref(x, y):
    return F.silu(x + y)

def estimate(params):
    N = params["N"]
    # x + y (N adds)
    # silu(z) (approx N ops)
    flops = 2 * N
    # 2 inputs, 1 output (3*N elements), assuming 2 bytes per element (fp16/bf16)
    bytes_ = 3 * N * 2
    return {"flops": flops, "bytes": bytes_}
