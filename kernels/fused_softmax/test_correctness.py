import torch


def test_fused_softmax_fp32():
    from kernels.fused_softmax.reference import make_inputs, ref
    from kernels.fused_softmax.triton_impl import run

    case = {"B": 4, "S": 512, "H": 64}
    inputs = make_inputs(case, device="cuda:0", seed=0, dtype=torch.float32)
    out_ref = ref(**inputs)
    out_tri = run(**inputs)
    torch.cuda.synchronize()

    assert torch.allclose(out_ref, out_tri, atol=1e-4, rtol=1e-4), (
        f"max diff = {(out_ref - out_tri).abs().max().item()}"
    )
    print("✅ fused_softmax fp32 correctness passed")


def test_fused_softmax_fp16():
    from kernels.fused_softmax.reference import make_inputs, ref
    from kernels.fused_softmax.triton_impl import run

    case = {"B": 4, "S": 512, "H": 64}
    inputs = make_inputs(case, device="cuda:0", seed=0, dtype=torch.float16)
    out_ref = ref(**inputs)
    out_tri = run(**inputs)
    torch.cuda.synchronize()

    assert torch.allclose(out_ref, out_tri, atol=1e-2, rtol=1e-2), (
        f"max diff = {(out_ref - out_tri).abs().max().item()}"
    )
    print("✅ fused_softmax fp16 correctness passed")


if __name__ == "__main__":
    test_fused_softmax_fp32()
    test_fused_softmax_fp16()
