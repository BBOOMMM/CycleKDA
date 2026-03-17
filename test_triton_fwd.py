import torch
import triton

# from myfla import chunk_kda, fused_recurrent_kda
from myfla.chunk_vecbeta import chunk_kda


def test_chunk_kda():
    torch.manual_seed(42)
    
    B, T, H, K, V = 2, 120, 4, 64, 64
    q = torch.randn(B, T, H, K).cuda()
    k = torch.randn(B, T, H, K).cuda()
    v = torch.randn(B, T, H, V).cuda()
    g = torch.randn(B, T, H, K).cuda()
    g = g - 2*g.max()
    # beta = torch.rand(B, T, H).sigmoid().cuda()
    beta = torch.rand(B, T, H, K).sigmoid().cuda()
    
    print("Testing pytorch...")
    o_pytorch, state_pytorch = chunk_kda(
                            q=q,
                            k=k,
                            v=v,
                            g=g,
                            beta=beta,
                            initial_state=None,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                            cu_seqlens=None,
                            initial_t=0,
                            T_cycle=8,
                            use_triton=False,
                        )
    print(f"Pytorch output shape: {o_pytorch.shape}")
    print(f"Pytorch output mean: {o_pytorch.mean().item():.6f}")
    
    print("\nTesting triton...")
    o_triton, state_triton = chunk_kda(
                            q=q,
                            k=k,
                            v=v,
                            g=g,
                            beta=beta,
                            initial_state=None,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                            cu_seqlens=None,
                            initial_t=0,
                            T_cycle=8,
                            use_triton=True,
                        )
    print(f"Triton output shape: {o_triton.shape}")
    print(f"Triton output mean: {o_triton.mean().item():.6f}")
    
    o_diff = (o_triton - o_pytorch).abs()
    print(f"\nOutput Max diff: {o_diff.max().item():.6f}")
    print(f"Output Mean diff: {o_diff.mean().item():.6f}")
    
    state_diff = (state_pytorch - state_triton).abs()
    print(f"\nState Max diff: {state_diff.max().item():.6f}")
    print(f"State Mean diff: {state_diff.mean().item():.6f}")
    
    for i in range(T):
        step_output_diff = (o_triton[:, i] - o_pytorch[:, i]).abs()
        if step_output_diff.max().item() > 1e-5:
            print(f"Step {i}: Output Max diff: {step_output_diff.max().item():.6f}, Mean diff: {step_output_diff.mean().item():.6f}")




@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[128, 256, 512, 1024, 2048],
        x_log=True,
        line_arg="provider",
        line_vals=["pytorch", "triton"],
        line_names=["PyTorch", "Triton"],
        styles=[("black", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="chunk_kda_fwd_ms",
        args={
            "B": 2,
            "H": 4,
            "K": 64,
            "V": 64,
            "dtype": torch.float16,
            "T_cycle": 64,
            "initial_t": 0,
            "use_qk_l2norm_in_kernel": True,
        },
    )
)
def benchmark_chunk_kda_fwd(T, provider, B, H, K, V, dtype, T_cycle, initial_t, use_qk_l2norm_in_kernel):
    # 要求 T 可被 64 整除（与你当前 Triton 前向实现一致）
    if T % 64 != 0:
        raise ValueError(f"T must be divisible by 64, got T={T}")

    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    g = g - 2 * g.max()
    # beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid()
    beta = torch.rand(B, T, H, K, device="cuda", dtype=dtype).sigmoid()

    def run():
        o, _ = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
            initial_t=initial_t,
            T_cycle=T_cycle,
            use_triton=(provider == "triton"),
        )
        return o

    # 返回 ms
    ms = triton.testing.do_bench(run, warmup=30, rep=200)
    return ms



if __name__ == "__main__":
    test_chunk_kda()
    benchmark_chunk_kda_fwd.run(save_path='.', print_data=False)