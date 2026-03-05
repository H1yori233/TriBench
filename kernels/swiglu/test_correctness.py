import torch


def test_swiglu_fp32():
    from kernels.swiglu.reference import make_inputs, ref
    from kernels.swiglu.triton_impl import run

    case = {"B": 1, "T": 1024, "H": 1024}
    inputs = make_inputs(case, device="cuda:0", seed=0, dtype=torch.float32)
    out_ref = ref(**inputs)
    out_tri = run(**inputs)
    torch.cuda.synchronize()

    assert torch.allclose(out_ref, out_tri, atol=1e-5, rtol=1e-4), (
        f"max diff = {(out_ref - out_tri).abs().max().item()}"
    )
    print("✅ swiglu fp32 correctness passed")


def test_swiglu_fp16():
    from kernels.swiglu.reference import make_inputs, ref
    from kernels.swiglu.triton_impl import run

    case = {"B": 1, "T": 1024, "H": 1024}
    inputs = make_inputs(case, device="cuda:0", seed=0, dtype=torch.float16)
    out_ref = ref(**inputs)
    out_tri = run(**inputs)
    torch.cuda.synchronize()

    assert torch.allclose(out_ref, out_tri, atol=1e-2, rtol=1e-2), (
        f"max diff = {(out_ref - out_tri).abs().max().item()}"
    )
    print("✅ swiglu fp16 correctness passed")


if __name__ == "__main__":
    test_swiglu_fp32()
    test_swiglu_fp16()
