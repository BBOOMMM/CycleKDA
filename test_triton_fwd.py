import torch
import triton

# from myfla import chunk_kda, fused_recurrent_kda
from myfla.chunk_vecbeta import chunk_kda


def test_chunk_kda(seed):
    torch.manual_seed(seed)
    
    B, T, H, K, V = 2, 120, 4, 64, 64
    q = torch.randn(B, T, H, K).cuda()
    k = torch.randn(B, T, H, K).cuda()
    v = torch.randn(B, T, H, V).cuda()
    g = torch.randn(B, T, H, K).cuda()
    g = g - 2*g.max()
    # beta = torch.rand(B, T, H).sigmoid().cuda()
    beta = torch.rand(B, T, H, K).sigmoid().cuda()
    
    # print("Testing pytorch...")
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
    # print(f"Pytorch output shape: {o_pytorch.shape}")
    # print(f"Pytorch output mean: {o_pytorch.mean().item():.6f}")
    
    # print("\nTesting triton...")
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
    # print(f"Triton output shape: {o_triton.shape}")
    # print(f"Triton output mean: {o_triton.mean().item():.6f}")
    
    o_diff = (o_triton - o_pytorch).abs()
    # print(f"\nOutput Max diff: {o_diff.max().item():.6f}")
    # print(f"Output Mean diff: {o_diff.mean().item():.6f}")
    
    state_diff = (state_pytorch - state_triton).abs()
    # print(f"\nState Max diff: {state_diff.max().item():.6f}")
    # print(f"State Mean diff: {state_diff.mean().item():.6f}")
    
    flag = True
    if o_diff.max().item() > 1e-5:
        print(f"Output Max diff: {o_diff.max().item():.6f}, Mean diff: {o_diff.mean().item():.6f}")
        flag = False
    
    if state_diff.max().item() > 1e-5:
        print(f"State Max diff: {state_diff.max().item():.6f}, Mean diff: {state_diff.mean().item():.6f}")
        flag = False
    
    for i in range(T):
        step_output_diff = (o_triton[:, i] - o_pytorch[:, i]).abs()
        if step_output_diff.max().item() > 1e-5:
            print(f"Step {i}: Output Max diff: {step_output_diff.max().item():.6f}, Mean diff: {step_output_diff.mean().item():.6f}")
            flag = False
    
    if flag:
        print(f"Test passed for seed {seed}!")
    else:
        print(f"Test failed for seed {seed}!")
        
        

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
        plot_name="chunk_kda_fwd_ms_vs_T",
        args={
            "B": 16,
            "H": 8,
            "K": 64,
            "V": 64,
            "dtype": torch.float16,
            "T_cycle": 16,
            "initial_t": 0,
            "use_qk_l2norm_in_kernel": True,
        },
    )
)
def benchmark_chunk_kda_fwd_vs_T(T, provider, B, H, K, V, dtype, T_cycle, initial_t, use_qk_l2norm_in_kernel):
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

    ms = triton.testing.do_bench(run)
    return ms



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[32, 64, 128, 256, 512, 1024],
        x_log=True,
        line_arg="provider",
        line_vals=["pytorch", "triton"],
        line_names=["PyTorch", "Triton"],
        styles=[("black", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="chunk_kda_fwd_ms_vs_B",
        args={
            "T": 128,
            "H": 8,
            "K": 64,
            "V": 64,
            "dtype": torch.float16,
            "T_cycle": 16,
            "initial_t": 0,
            "use_qk_l2norm_in_kernel": True,
        },
    )
)
def benchmark_chunk_kda_fwd_vs_B(T, provider, B, H, K, V, dtype, T_cycle, initial_t, use_qk_l2norm_in_kernel):
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

    ms = triton.testing.do_bench(run)
    return ms



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["K", "V"],
        x_vals=[(32,32), (64, 64), (128,128), (256,256)],
        x_log=True,
        line_arg="provider",
        line_vals=["pytorch", "triton"],
        line_names=["PyTorch", "Triton"],
        styles=[("black", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="chunk_kda_fwd_ms_vs_KV",
        args={
            "T": 128,
            "B": 16,
            "H": 8,
            "dtype": torch.float16,
            "T_cycle": 16,
            "initial_t": 0,
            "use_qk_l2norm_in_kernel": True,
        },
    )
)
def benchmark_chunk_kda_fwd_vs_KV(T, provider, B, H, K, V, dtype, T_cycle, initial_t, use_qk_l2norm_in_kernel):
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

    ms = triton.testing.do_bench(run)
    return ms




if __name__ == "__main__":
    for seed in range(5):
        test_chunk_kda(seed)
    benchmark_chunk_kda_fwd_vs_T.run(save_path='.', print_data=False)
    benchmark_chunk_kda_fwd_vs_B.run(save_path='.', print_data=False)
    benchmark_chunk_kda_fwd_vs_KV.run(save_path='.', print_data=False)