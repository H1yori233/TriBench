import torch


def test_rope_fp32():
    from kernels.rope.reference import make_inputs, ref
    from kernels.rope.triton_impl import run

    case = {"B": 1, "T": 512, "N_Q_H": 8, "N_KV_H": 2, "H": 64}
    inputs = make_inputs(case, device="cuda:0", seed=0, dtype=torch.float32)
    q_ref, k_ref = ref(**inputs)
    q_tri, k_tri = run(**inputs)
    torch.cuda.synchronize()

    assert torch.allclose(q_ref, q_tri, atol=1e-5, rtol=1e-4), (
        f"Q max diff = {(q_ref - q_tri).abs().max().item()}"
    )
    assert torch.allclose(k_ref, k_tri, atol=1e-5, rtol=1e-4), (
        f"K max diff = {(k_ref - k_tri).abs().max().item()}"
    )
    print("✅ rope fp32 correctness passed")


def test_rope_fp16():
    from kernels.rope.reference import make_inputs, ref
    from kernels.rope.triton_impl import run

    case = {"B": 1, "T": 512, "N_Q_H": 8, "N_KV_H": 2, "H": 64}
    inputs = make_inputs(case, device="cuda:0", seed=0, dtype=torch.float16)
    q_ref, k_ref = ref(**inputs)
    q_tri, k_tri = run(**inputs)
    torch.cuda.synchronize()

    assert torch.allclose(q_ref, q_tri, atol=1e-2, rtol=1e-2), (
        f"Q max diff = {(q_ref - q_tri).abs().max().item()}"
    )
    assert torch.allclose(k_ref, k_tri, atol=1e-2, rtol=1e-2), (
        f"K max diff = {(k_ref - k_tri).abs().max().item()}"
    )
    print("✅ rope fp16 correctness passed")


if __name__ == "__main__":
    test_rope_fp32()
    test_rope_fp16()
