import torch
import triton

# from myfla import chunk_kda, fused_recurrent_kda
from myfla.chunk_vecbeta import chunk_kda


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float):
    if a is None or b is None:
        raise AssertionError(f"{name}: got None (a={a is None}, b={b is None})")
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def test_chunk_kda_bwd(
    *,
    B: int = 2,
    T: int = 120,
    H: int = 4,
    K: int = 64,
    V: int = 64,
    dtype=torch.float16,
    T_cycle: int = 8,
    initial_t: int = 0,
    use_qk_l2norm_in_kernel: bool = True,
    include_state_in_loss: bool = False,
    seed: int = 0,
):
    torch.manual_seed(0)
    
    q0 = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    k0 = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    v0 = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g0 = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    g0 = g0 - 2 * g0.max()
    # beta0 = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid()
    beta0 = torch.rand(B, T, H, K, device="cuda", dtype=dtype).sigmoid()

    def run(provider: str):
        q = q0.clone().detach().requires_grad_(True)
        k = k0.clone().detach().requires_grad_(True)
        v = v0.clone().detach().requires_grad_(True)
        g = g0.clone().detach().requires_grad_(True)
        beta = beta0.clone().detach().requires_grad_(True)

        o, st = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
            initial_t=initial_t,
            T_cycle=T_cycle,
            use_triton=(provider == "triton"),
        )

        loss = o.float().mean()
        if include_state_in_loss:
            if st is None:
                raise AssertionError("include_state_in_loss=True but final_state is None")
            loss = loss + 0.1 * st.float().mean()

        # grads = torch.autograd.grad(loss, (q, k, v, g, beta), retain_graph=False, allow_unused=False)
        # dq, dk, dv, dg, dbeta = grads
        # return dq.detach(), dk.detach(), dv.detach(), dg.detach(), dbeta.detach()
        
        grads = torch.autograd.grad(
            loss, (q, k, v, g, beta),
            retain_graph=False,
            # allow_unused=True,   # <<< 改这里
        )
        dq, dk, dv, dg, dbeta = grads

        names = ["dq", "dk", "dv", "dg", "dbeta"]
        for n, gg in zip(names, grads):
            if gg is None:
                print(f"[{provider}] {n} is None (tensor not used in graph)")

        return (
            dq.detach() if dq is not None else None,
            dk.detach() if dk is not None else None,
            dv.detach() if dv is not None else None,
            dg.detach() if dg is not None else None,
            dbeta.detach() if dbeta is not None else None,
        )

    dq_p, dk_p, dv_p, dg_p, dbeta_p = run("pytorch")
    dq_t, dk_t, dv_t, dg_t, dbeta_t = run("triton")
    
    rtol = 1e-5
    atol = 1e-5

    _assert_close("dq", dq_t, dq_p, rtol=rtol, atol=atol)
    _assert_close("dk", dk_t, dk_p, rtol=rtol, atol=atol)
    _assert_close("dv", dv_t, dv_p, rtol=rtol, atol=atol)
    _assert_close("dg", dg_t, dg_p, rtol=rtol, atol=atol)
    _assert_close("dbeta", dbeta_t, dbeta_p, rtol=rtol, atol=atol)

    print(
        f"[BWD OK] T={T} dtype={dtype} "
        f"(rtol={rtol}, atol={atol}, include_state_in_loss={include_state_in_loss})"
    )



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
        plot_name="chunk_kda_bwd_ms_vs_T",
        args={
            "B": 16,
            "H": 8,
            "K": 64,
            "V": 64,
            "dtype": torch.float16,
            "T_cycle": 8,
            "initial_t": 0,
            "use_qk_l2norm_in_kernel": False,
            "include_state_in_loss": True,
        },
    )
)
def benchmark_chunk_kda_bwd_vs_T(
    T,
    provider,
    B,
    H,
    K,
    V,
    dtype,
    T_cycle,
    initial_t,
    use_qk_l2norm_in_kernel,
    include_state_in_loss,
):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    g = (g - 2 * g.max()).detach().requires_grad_(True)
    # beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid().detach().requires_grad_(True)
    beta = torch.rand(B, T, H, K, device="cuda", dtype=dtype).sigmoid().detach().requires_grad_(True)

    def run():
        q.grad = None
        k.grad = None
        v.grad = None
        g.grad = None
        beta.grad = None

        o, st = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
            initial_t=initial_t,
            T_cycle=T_cycle,
            use_triton=(provider == "triton"),
        )

        loss = o.float().mean()
        if include_state_in_loss:
            loss = loss + 0.1 * st.float().mean()

        loss.backward()

    ms = triton.testing.do_bench(run, warmup=30, rep=200)
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
        plot_name="chunk_kda_bwd_ms_vs_B",
        args={
            "T": 128,
            "H": 8,
            "K": 64,
            "V": 64,
            "dtype": torch.float16,
            "T_cycle": 8,
            "initial_t": 0,
            "use_qk_l2norm_in_kernel": False,
            "include_state_in_loss": True,
        },
    )
)
def benchmark_chunk_kda_bwd_vs_B(
    T,
    provider,
    B,
    H,
    K,
    V,
    dtype,
    T_cycle,
    initial_t,
    use_qk_l2norm_in_kernel,
    include_state_in_loss,
):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    g = (g - 2 * g.max()).detach().requires_grad_(True)
    # beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid().detach().requires_grad_(True)
    beta = torch.rand(B, T, H, K, device="cuda", dtype=dtype).sigmoid().detach().requires_grad_(True)

    def run():
        q.grad = None
        k.grad = None
        v.grad = None
        g.grad = None
        beta.grad = None

        o, st = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
            initial_t=initial_t,
            T_cycle=T_cycle,
            use_triton=(provider == "triton"),
        )

        loss = o.float().mean()
        if include_state_in_loss:
            loss = loss + 0.1 * st.float().mean()

        loss.backward()

    ms = triton.testing.do_bench(run, warmup=30, rep=200)
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
        plot_name="chunk_kda_bwd_ms_vs_KV",
        args={
            "T": 128,
            "B": 16,
            "H": 8,
            "dtype": torch.float16,
            "T_cycle": 8,
            "initial_t": 0,
            "use_qk_l2norm_in_kernel": False,
            "include_state_in_loss": True,
        },
    )
)
def benchmark_chunk_kda_bwd_vs_KV(
    T,
    provider,
    B,
    H,
    K,
    V,
    dtype,
    T_cycle,
    initial_t,
    use_qk_l2norm_in_kernel,
    include_state_in_loss,
):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn(B, T, H, K, device="cuda", dtype=dtype, requires_grad=True)
    g = (g - 2 * g.max()).detach().requires_grad_(True)
    # beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid().detach().requires_grad_(True)
    beta = torch.rand(B, T, H, K, device="cuda", dtype=dtype).sigmoid().detach().requires_grad_(True)

    def run():
        q.grad = None
        k.grad = None
        v.grad = None
        g.grad = None
        beta.grad = None

        o, st = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
            initial_t=initial_t,
            T_cycle=T_cycle,
            use_triton=(provider == "triton"),
        )

        loss = o.float().mean()
        if include_state_in_loss:
            loss = loss + 0.1 * st.float().mean()

        loss.backward()

    ms = triton.testing.do_bench(run, warmup=30, rep=200)
    return ms


if __name__ == "__main__":
    for seed in range(5):
        test_chunk_kda_bwd(T=120, dtype=torch.float16, include_state_in_loss=False, use_qk_l2norm_in_kernel=False, seed=seed)

    benchmark_chunk_kda_bwd_vs_B.run(save_path=".", print_data=False)
    benchmark_chunk_kda_bwd_vs_T.run(save_path=".", print_data=False)
    benchmark_chunk_kda_bwd_vs_KV.run(save_path=".", print_data=False)